from depth_projection.ref_depth_generator import dust3r_depth_generator
import argparse
import os

if __name__ == '__main__':
    # Generate Dust3r depth
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to image folder')
    parser.add_argument('--anchor', type=str, required=True, help='Anchor image filename (e.g., 20220819_104221). If not set, the first one will be used.')
    parser.add_argument('--n_dust3r', type=int, required=True, help='Number of images to generate multiview depth including anchor')
    parser.add_argument('--output', type=str, default='./output', help='Folder to save depth maps')
    parser.add_argument('--depth_anything', type=str, default=True, help='Generate monodepth from DepthAnythingv2')
    args = parser.parse_args()
    
    output_folder = "./output/" + args.folder.split("/")[-1]
    os.makedirs(output_folder, exist_ok=True)    

    dust3r_depth_generator(args.folder, args.anchor, args.n_dust3r, output_folder)