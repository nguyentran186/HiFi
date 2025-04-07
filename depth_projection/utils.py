import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import collections
import torchvision.utils as vutils
import struct
import math


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
ImageStruct = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def get_depth(points3D, K, c2w, img_size):
    """
    Computes a depth image by projecting 3D points onto a given camera view.

    Parameters:
    - points3D (torch.Tensor): (N, 3) tensor of 3D points in world coordinates (GPU).
    - K (torch.Tensor): (3, 3) camera intrinsic matrix (GPU).
    - c2w (torch.Tensor): (4, 4) camera-to-world transformation matrix (GPU).
    - img_size (tuple): (H, W) dimensions of the depth image.

    Returns:
    - depth_image (torch.Tensor): (H, W) depth image (GPU).
    """
    device = points3D.device

    # Compute world-to-camera transformation
    w2c = torch.inverse(c2w)  # (4,4)
    
    # Convert 3D points to homogeneous coordinates
    ones = torch.ones((points3D.shape[0], 1), device=device)
    points_hom = torch.cat([points3D, ones], dim=1)  # (N, 4)
    
    # Transform points to camera space
    cam_coords = (w2c @ points_hom.T).T  # (N, 4)
    depths = cam_coords[:, 2]  # Extract depth (Z values)
    
    # Project to 2D pixel coordinates
    xy_hom = (K @ cam_coords[:, :3].T).T  # (N, 3)
    xy_proj = xy_hom[:, :2] / xy_hom[:, 2:3]  # Normalize by depth: (N, 2)
    
    # Convert to integer pixel indices
    u = xy_proj[:, 0].long()
    v = xy_proj[:, 1].long()

    H, W = img_size
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths > 0)  # Filter valid points
    
    # Create depth image (initialize with large depth)
    depth_image = torch.full((H, W), float('inf'), device=device)

    # Populate depth image with the nearest depth values
    depth_image[v[valid], u[valid]] = torch.minimum(depth_image[v[valid], u[valid]], depths[valid])

    # Replace inf values with zero (occluded regions)
    depth_image[depth_image == float('inf')] = 0

    return depth_image

def read_colmap_points3D(bin_path, device="cuda"):
    """
    Reads a COLMAP points3D.bin file and extracts 3D points as a CUDA tensor.

    Parameters:
    - bin_path (str): Path to the COLMAP 'points3D.bin' file.
    - device (str): 'cuda' or 'cpu' (default: 'cuda').

    Returns:
    - points (torch.Tensor): (N, 3) tensor of 3D points stored in CUDA.
    """
    with open(bin_path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]  # Read number of points
        
        points = []
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]  # Read point ID
            xyz = struct.unpack("<ddd", f.read(24))  # Read X, Y, Z
            f.read(3)  # Skip RGB values (not used)
            f.read(8)  # Skip reprojection error
            
            track_length = struct.unpack("<Q", f.read(8))[0]  # Read track length
            f.read(track_length * 8)  # Skip image_id and point2D_idx for all tracks

            points.append(xyz)  # Append 3D point

    # Convert list to NumPy array first, then move to CUDA tensor
    points_np = np.array(points, dtype=np.float32)  # (N, 3)
    points_cuda = torch.tensor(points_np, dtype=torch.float32, device=device)  # Move to CUDA

    return points_cuda

def read_cameras_txt(filepath):
    """Read camera intrinsics from COLMAP's cameras.txt."""
    cameras = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = np.array(list(map(float, parts[4:])))
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params
            )
    return cameras

def read_images_txt(filepath):
    """Read camera extrinsics from COLMAP's images.txt."""
    images = {}
    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            image_id = int(parts[0])
            qvec = np.array(list(map(float, parts[1:5])))
            tvec = np.array(list(map(float, parts[5:8])))
            camera_id = int(parts[8])
            image_name = parts[9]
            next_line = f.readline().strip()
            elems = next_line.split()
            xys = np.column_stack([list(map(float, elems[0::3])),
                                   list(map(float, elems[1::3]))])
            point3D_ids = np.array(list(map(int, elems[2::3])))
            images[image_name] = ImageStruct(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids
            )
    return images

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
def get_intrinsics2(H, W, fovx, fovy):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def load_camera_info(sparse_folder):
    cameras_txt = os.path.join(sparse_folder, "cameras.txt")
    images_txt = os.path.join(sparse_folder, "images.txt")
    images_txt = os.path.join(sparse_folder, "images.txt")
    points_txt = os.path.join(sparse_folder, "points3D.bin")
    cameras = read_cameras_txt(cameras_txt)
    images = read_images_txt(images_txt)
    point3d = read_colmap_points3D(points_txt)

    cam_infos = {}
    for img_name, data in images.items():
        cam_data = cameras[1]
        width, height = cam_data.width, cam_data.height
        fx = cam_data.params[0]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fx, width)
        
        K = get_intrinsics2(height//4, width//4, FovX, FovY)
        
        R = qvec2rotmat(data.qvec).T
        T = -R @ np.array(data.tvec)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        
        cam_infos[img_name] = {
            "K": K,
            "c2w": c2w,
            "width": width,
            "height": height
        }
    return cam_infos, point3d

def compute_R_ts_T_ts(c2w_s, c2w_t):
    """
    Compute relative rotation and translation from source to target.

    Args:
        c2w_s: (4,4) source camera-to-world matrix (torch.Tensor, must be on CUDA)
        c2w_t: (4,4) target camera-to-world matrix (torch.Tensor, must be on CUDA)

    Returns:
        R_ts: (3,3) relative rotation matrix (torch.Tensor, on CUDA)
        T_ts: (3,) relative translation vector (torch.Tensor, on CUDA)
    """
    # Extract rotation (3x3) and translation (3x1) components
    R_s = c2w_s[:3, :3]
    T_s = c2w_s[:3, 3]
    R_t = c2w_t[:3, :3]
    T_t = c2w_t[:3, 3]

    # Compute relative rotation and translation
    R_ts = R_t.T @ R_s  # Equivalent to np.dot(R_t.T, R_s)
    T_ts = R_t.T @ (T_s - T_t)  # Equivalent to np.dot(R_t.T, (T_s - T_t))

    return R_ts, T_ts

def project_depth(depth, K, P_s, P_t):
    """
    Projects depth from source view to target view.

    Args:
        depth: (H, W) depth map from the source view (torch.Tensor, must be on CUDA)
        K: (3,3) camera intrinsic matrix (torch.Tensor, must be on CUDA)
        P_s: (3,4) source camera extrinsic matrix (torch.Tensor, must be on CUDA)
        P_t: (3,4) target camera extrinsic matrix (torch.Tensor, must be on CUDA)

    Returns:
        target_coords: (H, W, 2) pixel coordinates in target view (torch.Tensor, on CUDA)
    """
    device = depth.device
    H, W = depth.shape
    
    # Create meshgrid (H, W) and flatten
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x_normalized = (u - K[0, 2]) / K[0, 0]
    y_normalized = (v - K[1, 2]) / K[1, 1]
    
    pixels_homo = torch.stack([x_normalized*depth, y_normalized*depth, depth], dim=0).permute(1,2,0).reshape(-1, 3)  # (3, H*W)
    
    # R_ts, T_ts = compute_R_ts_T_ts( P_s, P_t)
    R_ts, T_ts = compute_R_ts_T_ts(P_t, P_s)
    
    X_3D_transformed = (R_ts @ pixels_homo.T + T_ts.reshape(3, 1)).T
    X_proj = K @ X_3D_transformed.T
    
    X_proj = (X_proj[:2] / (X_proj[2:3]))  # Normalize by depth
    # Reshape to (H, W, 2) with correct (x, y) order
    target_coords = X_proj.T.reshape(H, W, 2)

    return target_coords  # Output is still on CUDA

def gradient_loss(pred, target, mask):
    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]  # Remove extra dimension
    pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    target_dy = target[:, 1:, :] - target[:, :-1, :]

    # Pad gradients to match original shape
    pred_dx = F.pad(pred_dx, (0, 1, 0, 0))  # Pad width dimension
    pred_dy = F.pad(pred_dy, (0, 0, 0, 1))  # Pad height dimension
    target_dx = F.pad(target_dx, (0, 1, 0, 0))
    target_dy = F.pad(target_dy, (0, 0, 0, 1))

    loss_dx = torch.mean(((pred_dx - target_dx) ** 2) * mask)
    loss_dy = torch.mean(((pred_dy - target_dy) ** 2) * mask)

    return loss_dx + loss_dy

def warp_image(image, coords):
    """
    Warps image to target view using bilinear sampling.
    Args:
        image: (C, H, W) source image
        coords: (H, W, 2) target coordinates
    Returns:
        warped image (C, H, W)
    """
    H, W = image.shape[1:]
    coords = coords.clone()
    coords[..., 0] = (coords[..., 0] / (W - 1)) * 2 - 1  # Normalize to [-1,1]
    coords[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1
    warped = F.grid_sample(image.unsqueeze(0), coords.unsqueeze(0), mode='bilinear', align_corners=True)
    return warped.squeeze(0)

def total_variation_loss(depth, mask):
    """Computes TV loss only within the masked region."""
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)  # Add channel dimension (1, H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # Ensure mask has (1, H, W)

    dx = torch.abs(depth[:, :, :-1] - depth[:, :, 1:])
    dy = torch.abs(depth[:, :-1, :] - depth[:, 1:, :])

    # Apply mask to smooth only in masked regions
    masked_dx = dx * mask[:, :, :-1]
    masked_dy = dy * mask[:, :-1, :]

    return (masked_dx.mean() + masked_dy.mean())

def scale_invariant_mse_loss(pred_depth, target_depth, mask):
    """
    Compute the Scale-Invariant Mean Squared Error (SI-MSE) loss.

    Args:
        pred_depth (torch.Tensor): Predicted depth map (B, H, W).
        target_depth (torch.Tensor): Ground truth depth map (B, H, W).
        mask (torch.Tensor): Binary mask indicating valid depth pixels (B, H, W).

    Returns:
        torch.Tensor: Computed loss.
    """
    valid_mask = (target_depth > 0).float() * mask  # Ensure valid depth values
    
    breakpoint()
    k = valid_mask.sum(dim=[0, 1], keepdim=True)  # Number of valid pixels per image

    diff = (target_depth - pred_depth) * valid_mask  # Apply mask to errors
    diff_sq = diff ** 2

    first_term = diff_sq.sum(dim=[1, 0]) / k  # MSE term

    mean_diff = (diff.sum(dim=[1, 0], keepdim=True) / k)  # Mean error over valid pixels
    second_term = (mean_diff ** 2).sum(dim=[0, 1])  # Second term correction

    loss = first_term - second_term / k.squeeze()  # Compute SI-MSE Loss
    return loss.mean()  # Average over batch

def masked_l1_depth_loss(pred_depth, target_depth, mask):
    """
    Computes masked L1 loss between predicted and target depth maps, 
    considering only valid target depth values (nonzero).
    
    Parameters:
    - pred_depth (torch.Tensor): (H, W) Predicted depth map.
    - target_depth (torch.Tensor): (H, W) Ground truth depth map.
    - mask (torch.Tensor): (H, W) Binary mask (1 for valid regions, 0 otherwise).
    
    Returns:
    - loss (torch.Tensor): Scalar L1 loss value.
    """
    # Create a mask where target depth is nonzero
    valid_mask = (target_depth > 0).float()
    
    # Combine with the provided mask
    final_mask = mask * valid_mask
    # Compute L1 loss
    loss = torch.abs(pred_depth - target_depth)

    # Apply the mask and normalize
    return (loss * final_mask).sum() / (final_mask.sum() + 1e-6)

def masked_l1_loss(pred, target, mask):
    """Computes L1 loss only on valid mask regions."""
    loss = torch.abs(pred - target)
    # vutils.save_image(loss* mask, "a.png")
    # breakpoint()
    return (loss * mask).sum() / mask.sum()

def dilate_mask(mask, kernel_size=5):
    """Dilate a binary mask using max pooling."""
    pad = kernel_size // 2
    mask = mask.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
    dilated_mask = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    return dilated_mask.squeeze()

def dilate_mask_numpy(mask, kernel_size):
    """Dilates the input mask with the specified kernel size and converts it to a binary mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    _, binary_mask = cv2.threshold(dilated_mask, 128, 255, cv2.THRESH_BINARY)
    return binary_mask
