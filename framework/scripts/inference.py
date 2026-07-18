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

from multitask_project.utils import depth_to_rgb
from yolo_pipeline import (
    PostProcess,
    build_detection_model,
    draw_detections,
    load_class_map,
    read_mean_std,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Run HydraNet inference with segmentation, depth, and YOLO detections.")
    parser.add_argument("--input", default="data", help="Image file or directory of frames.")
    parser.add_argument("--seg-ckpt", default="checkpoints/ExpKITTI_joint.ckpt", help="Pretrained seg/depth checkpoint.")
    parser.add_argument("--det-ckpt", default=None, help="Optional detection head checkpoint.")
    parser.add_argument("--class-map-file", default=None, help="Optional class map definition file.")
    parser.add_argument("--cmap-file", default="data/cmap_kitti.npy", help="Semantic segmentation color map.")
    parser.add_argument("--mean-std-file", default="outputs/yolo_seg_depth/outputs/mean_std_kitti.txt")
    parser.add_argument("--output-video", default="outputs/videos/out.mp4")
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--seg-num-classes", type=int, default=6)
    parser.add_argument("--det-num-classes", type=int, default=14)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--iou-thres", type=float, default=0.3)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frames-dir", default="outputs/yolo_seg_depth/output/inference_frames")
    return parser


def _collect_inputs(path: str):
    in_path = Path(path)
    if in_path.is_dir():
        return sorted([p for p in in_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    if in_path.is_file():
        return [in_path]
    raise FileNotFoundError(path)


def _preprocess(image: np.ndarray, mean, std, image_height: int, image_width: int):
    resized = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - np.array(mean).reshape(1, 1, 3)) / np.array(std).reshape(1, 1, 3)
    resized = np.moveaxis(resized, -1, 0)
    return torch.from_numpy(resized).unsqueeze(0)


def run_inference(args):
    device = torch.device(args.device)
    mean, std = read_mean_std(args.mean_std_file)
    class_map = load_class_map(args.class_map_file)
    cmap = np.load(args.cmap_file)

    model = build_detection_model(
        seg_num_classes=args.seg_num_classes,
        det_num_classes=args.det_num_classes,
        seg_ckpt_path=args.seg_ckpt,
        det_ckpt_path=args.det_ckpt,
        device=device,
        reg_max=args.reg_max,
        train_backbone=False,
    )
    model.eval()
    postprocess = PostProcess(args.image_height, args.image_width, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

    input_paths = _collect_inputs(args.input)
    if not input_paths:
        raise RuntimeError("No input images found.")

    frames = []
    for path in input_paths:
        image = np.array(Image.open(path).convert("RGB"))
        original = cv2.resize(image, (args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
        tensor = _preprocess(image, mean, std, args.image_height, args.image_width).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            if len(outputs) == 3:
                segm, depth, det_out = outputs
            else:
                segm, depth = outputs[:2]
                det_out = None

        segm_np = segm[0, : args.seg_num_classes].cpu().numpy().transpose(1, 2, 0)
        segm_np = cv2.resize(segm_np, (args.image_width, args.image_height), interpolation=cv2.INTER_CUBIC)
        segm_color = (cmap[np.argmax(segm_np, axis=2)] * 255).astype(np.uint8)

        depth_np = depth[0, 0].cpu().numpy()
        depth_np = cv2.resize(depth_np, (args.image_width, args.image_height), interpolation=cv2.INTER_CUBIC)
        depth_rgb = depth_to_rgb(np.abs(depth_np))

        if det_out is not None:
            boxes, scores, class_ids = postprocess(det_out)
            det_frame = draw_detections(original.copy(), boxes, scores, class_ids, class_map)
        else:
            det_frame = original.copy()

        combined = cv2.vconcat([original, segm_color, depth_rgb, det_frame])
        frames.append(combined)

        if args.save_frames:
            save_dir = Path(args.frames_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"{path.stem}.jpg"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved inference video to {output_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
