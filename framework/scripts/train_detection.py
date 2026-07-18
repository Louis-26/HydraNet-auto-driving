from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from yolo_pipeline import (
    DetectionDataset,
    PostProcess,
    collate_detection,
    collect_pairs,
    evaluate_map50,
    load_class_map,
    make_detection_loss,
    build_detection_model,
    read_mean_std,
    save_checkpoint,
    train_one_epoch,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train the HydraNet YOLO detection head.")
    parser.add_argument("--train-image-dir", required=True, help="Directory containing training images.")
    parser.add_argument("--train-label-dir", required=True, help="Directory containing training label .txt files.")
    parser.add_argument("--val-image-dir", default=None, help="Directory containing validation images.")
    parser.add_argument("--val-label-dir", default=None, help="Directory containing validation label .txt files.")
    parser.add_argument("--seg-ckpt", default="checkpoints/ExpKITTI_joint.ckpt", help="Pretrained seg/depth checkpoint.")
    parser.add_argument("--det-ckpt", default=None, help="Optional detection-head checkpoint to resume from.")
    parser.add_argument("--class-map-file", default=None, help="Optional mapping file for detection classes.")
    parser.add_argument("--parser", choices=["kitti", "bdd100k"], default="kitti", help="Label parsing format.")
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--seg-num-classes", type=int, default=6)
    parser.add_argument("--det-num-classes", type=int, default=14)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/yolo_seg_depth/experiments/hydranet_od")
    parser.add_argument("--mean-std-file", default="outputs/yolo_seg_depth/outputs/mean_std_kitti.txt")
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--iou-thres", type=float, default=0.3)
    parser.add_argument("--train-backbone", action="store_true", help="Allow encoder/decoder fine-tuning.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--resume", default=None, help="Path to a saved training checkpoint.")
    return parser


def run_training(args):
    device = torch.device(args.device)
    mean, std = read_mean_std(args.mean_std_file)
    class_map = load_class_map(args.class_map_file)

    train_pairs = collect_pairs(args.train_image_dir, args.train_label_dir)
    if not train_pairs:
        raise RuntimeError("No training image/label pairs were found.")
    val_pairs = collect_pairs(args.val_image_dir, args.val_label_dir) if args.val_image_dir and args.val_label_dir else []

    train_dataset = DetectionDataset(
        train_pairs,
        image_height=args.image_height,
        image_width=args.image_width,
        class_map=class_map,
        parser=args.parser,
        mean=mean,
        std=std,
        train=True,
    )
    val_dataset = (
        DetectionDataset(
            val_pairs,
            image_height=args.image_height,
            image_width=args.image_width,
            class_map=class_map,
            parser=args.parser,
            mean=mean,
            std=std,
            train=False,
        )
        if val_pairs
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_detection,
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    model = build_detection_model(
        seg_num_classes=args.seg_num_classes,
        det_num_classes=args.det_num_classes,
        seg_ckpt_path=args.seg_ckpt,
        det_ckpt_path=args.det_ckpt,
        device=device,
        reg_max=args.reg_max,
        train_backbone=args.train_backbone,
    )
    criterion = make_detection_loss(
        image_height=args.image_height,
        image_width=args.image_width,
        det_num_classes=args.det_num_classes,
        stride=(8, 16, 32),
        reg_max=args.reg_max,
        device=device,
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    postprocess = PostProcess(args.image_height, args.image_width, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

    start_epoch = 0
    best_map = -1.0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        if "model_state_dict" in resume:
            model.load_state_dict(resume["model_state_dict"], strict=False)
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        if "scaler_state_dict" in resume and scaler.is_enabled():
            scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_map = float(resume.get("metrics", {}).get("mAP50", -1.0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_items = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            scheduler=scheduler,
        )
        print(
            f"train loss={train_loss:.4f} "
            f"box={train_items[0]:.4f} cls={train_items[1]:.4f} dfl={train_items[2]:.4f}"
        )

        metrics = {}
        if val_loader is not None:
            map50, ap_values = evaluate_map50(model, val_loader, postprocess, class_map, device)
            metrics = {"mAP50": map50, "AP50": ap_values}
            print(f"val mAP@0.5={map50:.4f}")
            if map50 > best_map:
                best_map = map50
                save_checkpoint(
                    output_dir / "best.pth",
                    model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    metrics=metrics,
                )

        save_checkpoint(
            output_dir / f"epoch_{epoch + 1:03d}.pth",
            model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            metrics=metrics,
        )

    print(f"Training complete. Outputs saved to {output_dir}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
