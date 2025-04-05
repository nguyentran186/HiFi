from depth_projection.ref_depth_generator import dust3r_depth_generator
from depth_projection.mono_depth_generator import run_depth_anything
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to image folder')
    parser.add_argument('--anchor', type=str, required=True, help='Anchor image filename (e.g., 20220819_104221). If not set, the first one will be used.')
    parser.add_argument('--n_dust3r', type=int, required=True, help='Number of images to generate multiview depth including anchor')
    parser.add_argument('--depth_anything', type=str, default=True, help='Generate monodepth from DepthAnythingv2')
    parser.add_argument('--output', type=str, default='./output', help='Folder to save depth maps')
    args = parser.parse_args()
    
    output_folder = "./output/" + args.folder.split("/")[-1]
    os.makedirs(output_folder, exist_ok=True)    

    anchor_image_path = os.path.join(args.folder, "images_4", f"{args.anchor}.png")
    depth_path = os.path.join(output_folder, "depth")
    
    # Generate Dust3r depth
    dust3r_depth_generator(args.folder, args.anchor, args.n_dust3r, depth_path)
    
    # Generate monodepth (DepthAnythingV2)
    run_depth_anything(anchor_image_path, depth_path)    