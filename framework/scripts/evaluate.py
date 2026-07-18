from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from yolo_pipeline import (
    DetectionDataset,
    PostProcess,
    build_detection_model,
    collate_detection,
    collect_pairs,
    evaluate_map50,
    load_class_map,
    read_mean_std,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the YOLO detection head on a labeled split.")
    parser.add_argument("--image-dir", required=True, help="Directory of images.")
    parser.add_argument("--label-dir", required=True, help="Directory of matching .txt labels.")
    parser.add_argument("--seg-ckpt", default="checkpoints/ExpKITTI_joint.ckpt", help="Pretrained seg/depth checkpoint.")
    parser.add_argument("--det-ckpt", default=None, help="Optional detection checkpoint.")
    parser.add_argument("--class-map-file", default=None, help="Optional class-map file.")
    parser.add_argument("--parser", choices=["kitti", "bdd100k"], default="kitti")
    parser.add_argument("--mean-std-file", default="outputs/yolo_seg_depth/outputs/mean_std_kitti.txt")
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--seg-num-classes", type=int, default=6)
    parser.add_argument("--det-num-classes", type=int, default=14)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--iou-thres", type=float, default=0.3)
    parser.add_argument("--output-json", default=None, help="Optional path to dump metrics as JSON.")
    return parser


def run(args):
    device = torch.device(args.device)
    mean, std = read_mean_std(args.mean_std_file)
    class_map = load_class_map(args.class_map_file)
    pairs = collect_pairs(args.image_dir, args.label_dir)
    if not pairs:
        raise RuntimeError("No image/label pairs were found for evaluation.")

    dataset = DetectionDataset(
        pairs,
        image_height=args.image_height,
        image_width=args.image_width,
        class_map=class_map,
        parser=args.parser,
        mean=mean,
        std=std,
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection,
    )

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

    map50, ap_values = evaluate_map50(model, loader, postprocess, class_map, device)
    result = {"mAP50": map50, "AP50": ap_values}
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
