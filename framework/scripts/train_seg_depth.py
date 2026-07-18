from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multitask_project.multitask_model import HydraNet
from multitask_project.utils import depth_to_rgb
from yolo_pipeline import load_state_dict_flexible, read_mean_std


def build_parser():
    parser = argparse.ArgumentParser(
        description="Stage-1 segmentation/depth compatibility entry point."
    )
    parser.add_argument("--checkpoint", default="checkpoints/ExpKITTI_joint.ckpt", help="Shared encoder/decoder weights.")
    parser.add_argument("--input", default=None, help="Optional folder of images for a quick sanity-check inference.")
    parser.add_argument("--output-dir", default="outputs/seg_depth_preview")
    parser.add_argument("--mean-std-file", default="outputs/yolo_seg_depth/outputs/mean_std_kitti.txt")
    parser.add_argument("--cmap-file", default="data/cmap_kitti.npy")
    parser.add_argument("--seg-num-classes", type=int, default=6)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _preprocess(image: np.ndarray, mean, std, image_height: int, image_width: int):
    resized = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - np.array(mean).reshape(1, 1, 3)) / np.array(std).reshape(1, 1, 3)
    resized = np.moveaxis(resized, -1, 0)
    return torch.from_numpy(resized).unsqueeze(0)


def run(args):
    if args.input is None:
        print(
            "Stage-1 seg/depth training is intentionally left as the historical compatibility path.\n"
            "The active cleaned pipeline focuses on YOLO detection head training via scripts/train_detection.py."
        )
        return

    device = torch.device(args.device)
    mean, std = read_mean_std(args.mean_std_file)
    cmap = np.load(args.cmap_file)

    model = HydraNet(num_tasks=2, num_classes=args.seg_num_classes)
    load_state_dict_flexible(model, args.checkpoint)
    model.to(device).eval()

    input_dir = Path(args.input)
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        image = np.array(Image.open(path).convert("RGB"))
        tensor = _preprocess(image, mean, std, args.image_height, args.image_width).to(device)
        with torch.no_grad():
            segm, depth = model(tensor)
        segm_np = segm[0, : args.seg_num_classes].cpu().numpy().transpose(1, 2, 0)
        segm_np = cv2.resize(segm_np, (args.image_width, args.image_height), interpolation=cv2.INTER_CUBIC)
        segm_color = (cmap[np.argmax(segm_np, axis=2)] * 255).astype(np.uint8)
        depth_np = depth_to_rgb(cv2.resize(depth[0, 0].cpu().numpy(), (args.image_width, args.image_height)))
        preview = cv2.vconcat([cv2.resize(image, (args.image_width, args.image_height)), segm_color, depth_np])
        cv2.imwrite(str(output_dir / f"{path.stem}.jpg"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    print(f"Saved seg/depth previews to {output_dir}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
