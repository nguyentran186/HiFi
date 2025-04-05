import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dust3r")) # adds HiFi/ to sys.path

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def compute_depth_map(pts3d, true_shape):
    if hasattr(pts3d, 'cpu'):
        pts3d = pts3d.cpu().numpy()
    H, W = true_shape
    depth_map = np.linalg.norm(pts3d.reshape(H, W, 3), axis=-1)
    return depth_map

def dust3r_depth_generator(folder_path, anchor_name, n, save_path='./depth_maps', device='cuda'):
    # Gather all image paths
    folder_path = os.path.join(folder_path, "images_4") 
    label_path = os.path.join(folder_path, "label")
    anchor_name = anchor_name + ".png"

    # Get images only from the label_path directory
    all_images = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_images = sorted(all_images)

    
    if anchor_name is None:
        anchor_name = all_images[0]
    
    if anchor_name not in all_images:
        raise ValueError(f"Anchor image '{anchor_name}' not found in {folder_path}")

    all_images.remove(anchor_name)
    if len(all_images) < n - 1:
        raise ValueError(f"Not enough images in folder to sample {n-1} others (only found {len(all_images)})")

    selected = random.sample(all_images, n - 1)
    selected.insert(0, anchor_name)  # Put anchor first

    full_paths = [os.path.join(folder_path, img) for img in selected]
    print(f"[INFO] Processing images: {selected}")

    # Load model
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    # Load images and infer
    images = load_images(full_paths, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)

    view1, pred1 = output['view1'], output['pred1']
    # Run global aligner
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
    depth_path = os.path.join(save_path, "depth")
    os.makedirs(depth_path, exist_ok=True)

    # Save depth map only for the anchor image
    for i in range(len(view1['img'])):
        instance_name = os.path.basename(view1['instance'][i])
        if instance_name == '0':
            print(f"[INFO] Saving dust3r depth map for anchor: {instance_name}")
            pts3d = pred1['pts3d'][i]
            true_shape = view1['true_shape'][i]
            depth = compute_depth_map(pts3d, true_shape)
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            anchor_name = anchor_name + folder_path.split('/')[-2]
            depth_png_path = os.path.join(depth_path, f"dust3r_depth.png")
            depth_npy_path = os.path.join(depth_path, f"dust3r_depth.npy")
            cv2.imwrite(depth_png_path, depth_norm)
            np.save(depth_npy_path, depth)
            break
    else:
        print("[WARNING] Anchor image not found in view1!")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to image folder')
    parser.add_argument('--anchor', type=str, default=None, help='Anchor image filename (e.g., 20220819_104221.png). If not set, a random one will be used.')
    parser.add_argument('--n', type=int, required=True, help='Number of images to process including anchor')
    parser.add_argument('--output', type=str, default='./depth_maps', help='Folder to save depth maps')
    args = parser.parse_args()

    process_folder(args.folder, args.anchor, args.n, save_path=args.output)

