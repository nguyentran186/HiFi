import os
import sys
import numpy as np
import collections
import cv2
from tqdm import tqdm

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
ImageStruct = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def get_intrinsics2(H, W, fovx, fovy):
    """Calculate camera intrinsic matrix."""
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])

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

def compute_R_ts_T_ts(c2w_s, c2w_t):
    """Compute relative rotation and translation."""
    R_s = c2w_s[:3, :3]
    T_s = c2w_s[:3, 3]
    R_t = c2w_t[:3, :3]
    T_t = c2w_t[:3, 3]
    R_ts = np.dot(R_t.T, R_s)
    T_ts = np.dot(R_t.T, (T_s - T_t))
    return R_ts, T_ts

def project_points(image, K, R, T, depth_map):
    """Project 2D image onto a new view using intrinsic and extrinsic parameters."""
    epsilon = 1e-6
    depth_map = np.where(depth_map == 0, epsilon, depth_map)

    height, width = image.shape[:2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x_normalized = (u - K[0, 2]) / K[0, 0]
    y_normalized = (v - K[1, 2]) / K[1, 1]
    X_3D = np.stack((x_normalized * depth_map, y_normalized * depth_map, depth_map), axis=-1)
    X_3D_flat = X_3D.reshape(-1, 3)
    X_3D_transformed = (R @ X_3D_flat.T + T.reshape(3, 1)).T
    X_projected = K @ X_3D_transformed.T
    points_2D = X_projected[:2, :] / X_projected[2, :]
    points_2D = points_2D.T

    projected_image = np.zeros_like(image)
    for i in range(points_2D.shape[0]):
        x, y = int(points_2D[i, 0]), int(points_2D[i, 1])
        if 0 <= x < width and 0 <= y < height:
            projected_image[y, x] = image.flat[i * 3:(i * 3) + 3]
    return projected_image

def dilate_mask(mask, kernel_size):
    """Dilates the input mask with the specified kernel size and converts it to a binary mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    _, binary_mask = cv2.threshold(dilated_mask, 128, 255, cv2.THRESH_BINARY)
    return binary_mask

import math
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

def depth_project(input_folder, output_folder, anchor_image, anchor_path, depth_path):
    """Process images with projection and blending."""
    
    # read info
    output_folder = os.path.join(output_folder, "images_4")
    mask_folder = os.path.join(input_folder, "images_4/label")
    image_folder = os.path.join(input_folder, "images_4")
    sparse_folder = os.path.join(input_folder, "sparse", "0")
    os.makedirs(output_folder, exist_ok=True)
    
    cameras_txt_path = os.path.join(sparse_folder, "cameras.txt")
    images_txt_path = os.path.join(sparse_folder, "images.txt")
    cameras = read_cameras_txt(cameras_txt_path)
    images = read_images_txt(images_txt_path)

    # read camera infos
    cam_infos = {}
    for img_name, data in images.items():   
        cam_data = cameras[1]
        width, height = cam_data.width, cam_data.height
        if cam_data.model == "PINHOLE":
            focal_length_x = cam_data.params[0]
            focal_length_y = cam_data.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif cam_data.model == "SIMPLE_PINHOLE":
            fx = fy = cam_data.params[0]
            cx, cy = cam_data.params[1:3]
        elif cam_data.model == "SIMPLE_RADIAL":
            fx = fy = cam_data.params[0]  # Focal length
            cx, cy = cam_data.params[1:3]  # Principal point offsets
            FovY = focal2fov(fx, height)
            FovX = focal2fov(fx, width)
        else:
            raise ValueError(f"Unsupported camera model: {cam_data['model']}")
        # K = get_intrinsics2(height, width, FovX, FovY)
        K = get_intrinsics2(height//4, width//4, FovX, FovY)

        # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
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
        
    # process anchor view
    anchor_data = cam_infos[f"{anchor_image}.jpg"]
    K_anchor = anchor_data['K']
    c2w_anchor = anchor_data['c2w']
    depth_anchor = np.load(depth_path)

    anchor_image_data = cv2.imread(anchor_path)
    temp_image_name = list(cam_infos.keys())[0]
    
    #for images_4 only
    # temp_shape = (cam_infos[temp_image_name]['height'], cam_infos[temp_image_name]['width'])
    temp_shape = (cam_infos[temp_image_name]['height']//4, cam_infos[temp_image_name]['width']//4)

    anchor_image_data = cv2.resize(anchor_image_data, (temp_shape[1], temp_shape[0]))
    depth_anchor = cv2.resize(depth_anchor, (temp_shape[1], temp_shape[0])) 
    
    output_path = os.path.join(output_folder, f"{anchor_image}.png")
    cv2.imwrite(output_path, anchor_image_data)
    
    
    # Project to other view
    for image_name, data in tqdm(cam_infos.items()):
        if image_name == f"{anchor_image}.jpg":
            continue
        image_name = image_name.replace('jpg', 'png')
        if image_name not in os.listdir(mask_folder):
            final_image = cv2.imread(os.path.join(image_folder, image_name))
        else:    
            K = data['K']
            c2w = data['c2w']
            R_ts, T_ts = compute_R_ts_T_ts(c2w_anchor, c2w)

            image_path = os.path.join(image_folder, image_name)
            original_image = cv2.imread(image_path)
            projected_image = project_points(anchor_image_data, K_anchor, R_ts, T_ts, depth_anchor)
            mask_path = os.path.join(mask_folder, image_name.replace("jpg", "png"))
            if os.path.exists(mask_path):
                # Load and preprocess mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) * 255
                mask = cv2.resize(mask, (temp_shape[1], temp_shape[0]))
                mask = dilate_mask(mask, kernel_size=0)
                
                # Create inner and outer masks
                mask_outer = dilate_mask(mask, kernel_size=50)  # Outer boundary
                mask_inner = dilate_mask(mask, kernel_size=20)  # Inner boundary

                # Compute boundary region (outer - inner)
                boundary_region = cv2.bitwise_xor(mask_outer, mask_inner)

                # Convert masks to binary
                mask_outer = (mask_outer > 0).astype(np.uint8)
                mask_inner = (mask_inner > 0).astype(np.uint8)
                boundary_mask = (boundary_region > 0).astype(np.uint8)  # Mask for blending

                # ---- Compute Blending Weights ----
                # Distance from outer boundary to inner
                dist_transform = cv2.distanceTransform(mask_outer, cv2.DIST_L2, 5)
                
                # Normalize blend weight to range [0,1] (avoid divide-by-zero)
                max_distance = dist_transform[boundary_mask > 0].max() if np.any(boundary_mask > 0) else 1
                blend_weight = (dist_transform / max_distance).clip(0, 1)
                
                # Ensure blending applies only in boundary
                blend_weight *= boundary_mask  

                # Expand dims for broadcasting (H, W) → (H, W, 1)
                blend_weight = np.expand_dims(blend_weight, axis=-1)

                # ---- Apply Weighted Blending in Boundary ----
                blended_boundary = (original_image * (1 - blend_weight) + projected_image * blend_weight).astype(np.uint8)

                # ---- Construct Final Image ----
                final_image = np.where(mask_outer[..., None] > 0, projected_image, original_image)  # Inside: Projected, Outside: Original
                final_image = np.where(boundary_mask[..., None] > 0, blended_boundary, final_image)  # Apply blended region

                # Ensure `projected_image` is checked per-channel
                black_pixel_mask = np.all(projected_image == 0, axis=-1, keepdims=True)  # Find fully black pixels in projected_image
                final_image = np.where((mask_outer[..., None] > 0) & black_pixel_mask, 0, final_image)  # Apply only in outer masked area

        
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, final_image)


