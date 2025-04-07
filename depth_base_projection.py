from depth_projection.ref_depth_generator import dust3r_depth_generator
from depth_projection.mono_depth_generator import run_depth_anything
from depth_projection.world_coord_depth_generator import world_coor_depth
from depth_projection.combine_depth import combine_depth
from depth_projection.depth_projection import depth_project
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to image folder')
    parser.add_argument('--anchor', type=str, required=True, help='Anchor image filename (e.g., 20220819_104221). If not set, the first one will be used.')
    parser.add_argument('--skip_depth_prepare', action='store_true', help='Skip the step preparing depth')
    parser.add_argument('--skip_depthgen', action='store_true', help='Skip the step generating depth')
    parser.add_argument('--output', type=str, default='./output', help='Folder to save depth maps')
    args = parser.parse_args()
    
    dataset_name = args.folder.split("/")[-1]
    output_folder =  os.path.join("output", dataset_name)
    inpainted_path = os.path.join("refs", f"{dataset_name}_out.png")
    os.makedirs(output_folder, exist_ok=True)    

    anchor_image_path = os.path.join(args.folder, "images_4", f"{args.anchor}.png")
    depth_path = os.path.join(output_folder, "depth")
    
    if not args.skip_depth_prepare:
        if not args.skip_depthgen:
            print(f"[INFO] Generating Dust3r depth for {args.anchor}...")
            # Generate Dust3r depth
            dust3r_depth_generator(args.folder, args.anchor, depth_path)

            print(f"[INFO] Generating monodepth (DepthAnythingV2) for {args.anchor}...")
            # Generate monodepth (DepthAnythingV2)
            run_depth_anything(inpainted_path, depth_path)
        
            print(f"[INFO] Generating combined depth map for {args.anchor}...")
            # Generate combine_depth
            combine_depth(args.folder, args.anchor, output_folder)
        
        print(f"[INFO] Regressing depth to world coordinates...")
        # Regress depth to world coordinates    
        mono_depth_path = os.path.join(depth_path, "combined_depth.npy")
        world_coor_depth(args.folder, args.anchor, mono_depth_path, output_folder)
    
    else:
        print("Skipping depth preparation steps...")

    # Depth base projection
    print(f"[INFO] Projecting depth to world coordinates")
    aligned_depth_path = os.path.join(depth_path, "aligned_depth.npy")
    depth_project(args.folder, output_folder, args.anchor, inpainted_path, aligned_depth_path)

    print("Process complete.")