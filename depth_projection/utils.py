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

