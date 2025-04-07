import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "depth_projection"))
from utils import load_camera_info, get_depth, masked_l1_depth_loss
import matplotlib.pyplot as plt

input_dim = 9

class DepthCorrectionMLP(nn.Module):
    def __init__(self, kernel_size=5):
        super(DepthCorrectionMLP, self).__init__()
        input_dim = kernel_size * kernel_size  # Number of depth values in a kernel
        self.fc1 = nn.Linear(input_dim, 64)  
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output corrected depth for center pixel

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output corrected depth


def get_depth_mlp(depth, mlp):
    """
    Get the adjusted depth from the MLP by extracting patches from the depth map.
    """
    patches, H, W = extract_patches(depth, kernel_size=input_dim)

    # Pass patches through MLP
    adjusted_patches = mlp(patches)  # Output shape: (num_patches, 1)

    # Reshape correctly for the image dimensions
    adjusted_depth = adjusted_patches.view(H, W)  # Adjust for kernel_size=5
    
    return adjusted_depth


def extract_patches(depth, kernel_size=5):
    """
    Extracts kernel-wise patches from a single-channel depth map.
    Returns: (num_patches, kernel_size * kernel_size), H, W
    """
    H, W = depth.shape
    pad_size = kernel_size // 2

    depth = depth.unsqueeze(0).unsqueeze(0)  # Adding batch and channel dimension
    depth_padded = F.pad(depth, (pad_size, pad_size, pad_size, pad_size), mode="replicate")

    patches = F.unfold(depth_padded, kernel_size=(kernel_size, kernel_size), stride=1)
    patches = patches.squeeze(0).T  # Transpose for correct shape
    
    return patches, H, W


def train_mlp_on_depth(depth, target_depth, mask, mlp, optimizer, output_dir, num_epochs=1500):
    """
    Training loop for DepthCorrectionMLP with a progress bar and loss visualization. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = mlp.to(device)
    depth = depth.to(device)
    target_depth = target_depth.to(device)

    # Initialize the progress bar with the number of epochs
    pbar = tqdm(range(num_epochs), desc="Training Epochs", ncols=100)

    # Store losses for plotting
    loss_list = []

    for epoch in pbar:
        mlp.train()
        optimizer.zero_grad()

        # Adjust depth using the MLP
        adjusted_depth = get_depth_mlp(depth, mlp)

        # Calculate loss
        loss = masked_l1_depth_loss(adjusted_depth, target_depth, mask)

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        # Update the progress bar with the current loss
        pbar.set_postfix(loss=loss.item())

        # Append the loss to the list for visualization
        loss_list.append(loss.item())

    # After training, visualize and save the loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(num_epochs), loss_list, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG image
    output_dir = os.path.join(output_dir, 'visualize')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # Return the final adjusted depth
    return adjusted_depth


# --- Training Loop --- 
def world_coor_depth(
    input_folder, anchor_view, depth_path,
    output_dir, dilation_kernel_size=150
):
    """
    Modify the previous training loop to integrate MLP-based depth correction and save results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_folder = os.path.join(input_folder, "sparse", "0")
    anchor_view = anchor_view + ".jpg"
    # Initialize MLP and optimizer
    mlp = DepthCorrectionMLP(kernel_size=input_dim).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)

    # Load camera info 
    cam_infos, points3d = load_camera_info(sparse_folder)
    anchor_info = cam_infos[anchor_view]
    K = torch.tensor(anchor_info['K'], dtype=torch.float32, device=device)
    P_s = torch.tensor(anchor_info['c2w'][:3], dtype=torch.float32, device=device)

    # Load depth
    depth = np.load(depth_path)
    depth = torch.tensor(depth, dtype=torch.float32, requires_grad=True, device=device)
    
    if len(depth.shape) == 3:
        depth = depth.mean(dim=2)
    
    # Load and process mask
    mask_dir = os.path.join(input_folder, "images_4", "label")
    original_mask = cv2.imread(os.path.join(mask_dir, anchor_view.replace("jpg", "png")), cv2.IMREAD_GRAYSCALE)
    original_mask = cv2.resize(original_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    kernel_large = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    kernel_small = np.ones((20, 20), np.uint8)
    
    mask_inner = cv2.dilate(original_mask, kernel_small, iterations=1)
    mask_outer = cv2.dilate(original_mask, kernel_large, iterations=1)
    mask = torch.tensor(mask_outer * (1 - mask_inner), dtype=torch.float32, device=device)
    # Get target depth
    tgt_depth = get_depth(points3d, K, torch.tensor(anchor_info['c2w'], dtype=torch.float32, device=device), depth.shape)
    
    adjusted_depth = train_mlp_on_depth(depth, tgt_depth, mask, mlp, optimizer, output_dir)
    adjusted_depth = adjusted_depth.detach().cpu().numpy()
    
    depth_npy_path = os.path.join(output_dir, "depth", "aligned_depth.npy")
    depth_img_path = os.path.join(output_dir, "depth", "aligned_depth.png")
    np.save(depth_npy_path, adjusted_depth)
    cv2.imwrite(depth_img_path, adjusted_depth)
    
    