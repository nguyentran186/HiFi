import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
import os
import cv2
from torchvision.io import read_image
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

# ------------------------------
# Dilate the mask (Optional, you can adjust as needed)
# ------------------------------
def dilate_mask(mask, kernel_size=50):
    structure = torch.ones((kernel_size, kernel_size), dtype=bool)
    dilated = binary_dilation(mask.squeeze().cpu().numpy(), structure=structure)
    return torch.from_numpy(dilated).to(mask.device).unsqueeze(0).unsqueeze(0).bool()

# ------------------------------
# Resize tensor to match size
# ------------------------------
def resize_to(tensor, target_tensor):
    if isinstance(target_tensor, torch.Tensor):
        _, _, H, W = target_tensor.shape
    else:
        raise ValueError("target_tensor must be a torch.Tensor with shape (B, C, H, W)")
    return F.interpolate(tensor.float(), size=(H, W), mode='bilinear', align_corners=False)

# ------------------------------
# Define MLP (Neural network model)
# ------------------------------
class DepthMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        input_dim = 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)

# ------------------------------
# Load tensor from image path
# ------------------------------
def load_tensor(path):
    # Check if the file is a .npy file
    if path.endswith(".npy"):
        # Load the .npy file
        npy_data = np.load(path)
        # Convert to tensor (make sure it's float32 for depth data)
        tensor = torch.tensor(npy_data, dtype=torch.float32)
        
        if tensor.ndimension() == 3:
            tensor = tensor.mean(dim=2)
        
        # Ensure it has a channel dimension (H, W) -> (1, H, W)
        if tensor.ndimension() == 2:  # single-channel image (depth map)
            tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor  # (1, H, W) or (C, H, W)
    
    else:
        # Use torchvision to read the image and normalize it
        tensor = read_image(path).float() / 255.0
        # Handle single channel or RGB
        if tensor.shape[0] == 1:
            return tensor.unsqueeze(0)  # (1, 1, H, W)
        else:
            return tensor[:1].unsqueeze(0)  # use first channel only (assuming grayscale)

# ------------------------------
# Inference and fuse depth (use mono depth to fill the mask area in duster depth)
# ------------------------------
def infer_and_fuse(depth_mono, model, save_path):
    model.eval()
    with torch.no_grad():
        # Ensure depth_mono and mask have the shape (B, C, H, W)
        depth_mono = depth_mono.unsqueeze(0) if depth_mono.dim() == 3 else depth_mono

        # Flatten depth_mono to pass to the model
        pred_depth = model(depth_mono.view(1, 1, -1).transpose(1, 2)).view(1, 1, *depth_mono.shape[2:])

        pred_depth = (pred_depth.squeeze()).cpu().numpy()
        np.save(save_path, pred_depth)

# ------------------------------
# Training step (Single step for model)
# ------------------------------
def train_step(depth_mono, depth_duster, mask, model, optimizer, criterion):
    model.train()

    # Resize and prepare masks
    mask = resize_to(mask, depth_duster).bool()
    surround_mask = ~mask  # area outside the mask

    # Extract valid surround pixels (outside the mask)
    src_pixels = depth_mono.squeeze()[surround_mask.squeeze()].unsqueeze(1)
    tgt_pixels = depth_duster.squeeze()[surround_mask.squeeze()].unsqueeze(1)

    # Forward pass (Model predicts values based on mono depth)
    preds = model(src_pixels)
    loss = criterion(preds, tgt_pixels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ------------------------------
# Train for multiple epochs
# ------------------------------
def combine_depth(input_folder, anchor, output_folder, num_epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    depth_path = os.path.join(output_folder, "depth")
    
    depth_mono = load_tensor(os.path.join(depth_path, "mono_depth.npy")).to(device)
    depth_duster = load_tensor(os.path.join(depth_path, "dust3r_depth.npy")).to(device)
    
    mask_path = os.path.join(input_folder, "images_4", "label", f'{anchor}.png')
    mask = (load_tensor(mask_path) * 255) > 0.5
    mask = resize_to(mask, depth_mono).to(device)
    depth_duster = resize_to(depth_duster, depth_mono).to(device)

    # Use tqdm for progress bar over epochs
    with tqdm(range(num_epochs), desc="Training Epochs") as pbar:
        for epoch in pbar:
            # Train for one step
            loss = train_step(depth_mono, depth_duster, mask, model, optimizer, criterion)
            pbar.set_postfix({"Loss": f"{loss:.4f}"})  # Update progress bar with loss

    # Inference and save prediction after final epoch
    save_path = os.path.join(depth_path, f"combined_depth.npy")
    infer_and_fuse(depth_mono.squeeze(0), model, save_path)
