import os
import glob
import cv2
import torch
import numpy as np
import matplotlib

import sys
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Depth-Anything-V2")))
from depth_anything_v2.dpt import DepthAnythingV2


def run_depth_anything(img_path: str, outdir: str):
    input_size = 518
    encoder = 'vitl'
    pred_only = True
    grayscale = True

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Load model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get filenames
    if os.path.isfile(img_path):
        if img_path.endswith('.txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)

    os.makedirs(outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f'Could not read image {filename}, skipping...')
            continue

        # Infer depth
        depth = depth_anything.infer_image(raw_image, input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Apply colormap or grayscale
        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Save result
        output_filename = os.path.join(outdir,'mono_depth.npy')
        if pred_only:
            np.save(output_filename, depth)
