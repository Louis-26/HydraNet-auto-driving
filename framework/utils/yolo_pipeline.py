from __future__ import annotations

import ast
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multitask_project.heads import DetectionHead, DetectionLoss, xywh2xyxy
from multitask_project.multitask_model import HydraNet


DEFAULT_CLASS_MAP = {
    "Car": 0,
    "Pedestrian": 1,
    "Van": 2,
    "Cyclist": 3,
    "Truck": 4,
    "Tram": 5,
    "Person_sitting": 6,
    "Rider": 7,
    "Bus": 8,
    "Train": 9,
    "Motorcycle": 10,
    "Bicycle": 11,
    "Traffic-sign": 12,
    "Traffic-light": 13,
}

DEFAULT_IGNORE_CLASSES = ("DontCare", "Misc")
DEFAULT_IMAGE_SIZE = (192, 640)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def read_mean_std(path: Optional[str]) -> Tuple[List[float], List[float]]:
    if not path:
        return DEFAULT_MEAN, DEFAULT_STD
    file_path = Path(path)
    if not file_path.exists():
        return DEFAULT_MEAN, DEFAULT_STD
    lines = file_path.read_text(encoding="utf-8").splitlines()
    mean = ast.literal_eval(" ".join(lines[0].split()[1:]))
    std = ast.literal_eval(" ".join(lines[1].split()[1:]))
    return mean, std


def load_class_map(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return DEFAULT_CLASS_MAP.copy()
    file_path = Path(path)
    if file_path.suffix.lower() in {".json", ".js"}:
        return {str(k): int(v) for k, v in json.loads(file_path.read_text(encoding="utf-8")).items()}
    mapping = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        mapping[key] = int(value)
    return mapping


def build_detection_transforms(
    image_height: int,
    image_width: int,
    mean: Sequence[float],
    std: Sequence[float],
    train: bool,
):
    transforms = [A.Resize(image_height, image_width)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.3)
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ]
    )
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.10,
        ),
    )


def collect_pairs(image_dir: str, label_dir: str, image_suffixes: Sequence[str] = (".jpg", ".jpeg", ".png")):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in image_suffixes])
    pairs = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((str(image_path), str(label_path)))
    return pairs


def _parse_kitti_lines(lines, class_map, ignore_classes):
    items = []
    for line in lines:
        parts = line.split()
        if not parts or parts[0] in ignore_classes:
            continue
        if parts[0] not in class_map:
            continue
        x1, y1, x2, y2 = map(float, parts[4:8])
        items.append((class_map[parts[0]], [x1, y1, x2, y2]))
    return items


def _parse_bdd_lines(lines, class_map):
    items = []
    for line in lines:
        parts = line.split()
        if not parts or parts[0] not in class_map:
            continue
        x1, y1, x2, y2 = map(float, parts[2:6])
        items.append((class_map[parts[0]], [x1, y1, x2, y2]))
    return items


class DetectionDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        image_height: int,
        image_width: int,
        class_map: Optional[Dict[str, int]] = None,
        parser: str = "kitti",
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        train: bool = True,
        ignore_classes: Sequence[str] = DEFAULT_IGNORE_CLASSES,
        max_resample: int = 12,
    ):
        self.pairs = list(pairs)
        self.image_height = image_height
        self.image_width = image_width
        self.class_map = class_map or DEFAULT_CLASS_MAP.copy()
        self.parser = parser
        self.ignore_classes = set(ignore_classes)
        self.max_resample = max_resample
        self.final_transform = build_detection_transforms(
            image_height=image_height,
            image_width=image_width,
            mean=mean or DEFAULT_MEAN,
            std=std or DEFAULT_STD,
            train=train,
        )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read_image(path: str):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_label(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.read().splitlines() if line.strip()]

    def _parse_labels(self, lines):
        if self.parser.lower() == "bdd100k":
            return _parse_bdd_lines(lines, self.class_map)
        return _parse_kitti_lines(lines, self.class_map, self.ignore_classes)

    def _normalize_targets(self, bboxes):
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            x_center = (x1 + x2) / 2 / self.image_width
            y_center = (y1 + y2) / 2 / self.image_height
            width = (x2 - x1) / self.image_width
            height = (y2 - y1) / self.image_height
            bboxes[i] = [x_center, y_center, width, height]

    def __getitem__(self, idx):
        for _ in range(self.max_resample):
            image_path, label_path = self.pairs[idx]
            image = self._read_image(image_path)
            labels = self._read_label(label_path)
            parsed = self._parse_labels(labels)
            if not parsed:
                idx = np.random.randint(0, len(self.pairs))
                continue

            class_ids = [cls for cls, _ in parsed]
            bboxes = [box for _, box in parsed]
            aug = self.final_transform(image=image, bboxes=bboxes, category_ids=class_ids)
            image = aug["image"]
            bboxes = list(aug["bboxes"])
            class_ids = list(aug["category_ids"])
            if not bboxes:
                idx = np.random.randint(0, len(self.pairs))
                continue

            self._normalize_targets(bboxes)
            target = torch.column_stack(
                [
                    torch.tensor(class_ids, dtype=torch.float32),
                    torch.tensor(bboxes, dtype=torch.float32),
                ]
            )
            return image, target
        raise RuntimeError(f"Could not sample a valid detection target from index {idx}")


def collate_detection(batch):
    images, targets = zip(*batch)
    stacked_images = torch.stack(images, dim=0)
    concatenated_targets = torch.cat(
        [torch.cat([idx * torch.ones(target.size(0), 1), target], dim=1) for idx, target in enumerate(targets)],
        dim=0,
    )
    return stacked_images, concatenated_targets


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area + 1e-9
    return intersection_area / union_area


def nms_single_class(boxes, scores, iou_threshold):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]
    return keep


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    keep_boxes = []
    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        if class_indices.size == 0:
            continue
        class_keep = nms_single_class(boxes[class_indices], scores[class_indices], iou_threshold)
        keep_boxes.extend(class_indices[class_keep].tolist())
    return keep_boxes


class PostProcess:
    def __init__(self, image_height, image_width, conf_thres=0.3, iou_thres=0.3):
        self.image_height = image_height
        self.image_width = image_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.detach().cpu().numpy()
        boxes_list, scores_list, class_ids_list = [], [], []
        for output in outputs:
            predictions = output.T
            if predictions.shape[1] < 5:
                boxes_list.append([])
                scores_list.append([])
                class_ids_list.append([])
                continue
            scores = np.max(predictions[:, 4:], axis=1)
            keep = scores > self.conf_thres
            predictions = predictions[keep]
            scores = scores[keep]
            if len(scores) == 0:
                boxes_list.append([])
                scores_list.append([])
                class_ids_list.append([])
                continue

            class_ids = np.argmax(predictions[:, 4:], axis=1)
            boxes = xywh2xyxy(predictions[:, :4])
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.image_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.image_height)
            indices = multiclass_nms(boxes, scores, class_ids, self.iou_thres)
            boxes_list.append(boxes[indices])
            scores_list.append(scores[indices])
            class_ids_list.append(class_ids[indices])

        if len(boxes_list) == 1:
            return boxes_list[0], scores_list[0], class_ids_list[0]
        return boxes_list, scores_list, class_ids_list


def _color_for_class(class_id: int):
    palette = [
        (255, 99, 71),
        (0, 191, 255),
        (50, 205, 50),
        (255, 215, 0),
        (138, 43, 226),
        (255, 140, 0),
        (255, 105, 180),
        (64, 224, 208),
        (255, 69, 0),
        (154, 205, 50),
    ]
    return palette[class_id % len(palette)]


def draw_detections(image, boxes, scores, class_ids, class_map):
    if boxes is None or len(boxes) == 0:
        return image

    annotated = image.copy()
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    class_ids = np.asarray(class_ids)

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int).tolist()
        color = _color_for_class(int(class_id))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_map.get(int(class_id), str(class_id))} {float(score):.2f}"
        text_y = max(y1 - 8, 12)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return annotated


def load_state_dict_flexible(model, checkpoint_path: str, key_candidates=("model_state_dict", "state_dict", "model", "weights")):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in key_candidates:
            if key in checkpoint:
                return model.load_state_dict(checkpoint[key], strict=False)
    return model.load_state_dict(checkpoint, strict=False)


def build_detection_model(
    seg_num_classes: int,
    det_num_classes: int,
    seg_ckpt_path: str,
    det_ckpt_path: Optional[str],
    device: torch.device,
    head_channels: Tuple[int, int, int] = (64, 128, 256),
    decoder_channels: Tuple[int, int, int] = (256, 256, 256),
    stride: Tuple[int, int, int] = (8, 16, 32),
    reg_max: int = 16,
    train_backbone: bool = False,
):
    model = HydraNet(num_tasks=2, num_classes=seg_num_classes)
    load_state_dict_flexible(model, seg_ckpt_path)
    detection_head = DetectionHead(
        num_classes=det_num_classes,
        decoder_channels=decoder_channels,
        head_channels=head_channels,
        stride=stride,
        reg_max=reg_max,
    )
    model.attach_detection_head(detection_head)
    if det_ckpt_path:
        load_state_dict_flexible(model.detect_head, det_ckpt_path)

    if not train_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = "detect_head" in name

    model.to(device)
    return model


def make_detection_loss(
    image_height: int,
    image_width: int,
    det_num_classes: int,
    stride: Tuple[int, int, int],
    reg_max: int,
    device: torch.device,
):
    return DetectionLoss(
        image_height=image_height,
        image_width=image_width,
        num_classes=det_num_classes,
        stride=stride,
        reg_max=reg_max,
        device=device,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler: Optional[GradScaler] = None,
    scheduler=None,
    log_interval: int = 20,
):
    model.train()
    running_loss = 0.0
    running_items = np.zeros(3, dtype=np.float64)
    amp_enabled = scaler is not None and scaler.is_enabled()

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            det_outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            loss, loss_items = criterion(det_outputs, targets)

        if amp_enabled:
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scale_before <= scaler.get_scale():
                scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        running_loss += float(loss.item())
        running_items += loss_items.detach().cpu().numpy()
        if step % log_interval == 0 or step == len(loader):
            print(
                f"step {step:04d}/{len(loader):04d} "
                f"loss={running_loss / step:.4f} "
                f"box={running_items[0] / step:.4f} "
                f"cls={running_items[1] / step:.4f} "
                f"dfl={running_items[2] / step:.4f}"
            )

    return running_loss / max(len(loader), 1), running_items / max(len(loader), 1)


def _average_precision(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map50(model, loader, postprocess: PostProcess, class_map: Dict[str, int], device):
    model.eval()
    id_to_name = {v: k for k, v in class_map.items()}
    per_class_preds = {cid: [] for cid in id_to_name}
    per_class_gts = {cid: {} for cid in id_to_name}

    with torch.no_grad():
        sample_offset = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            det_outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            boxes_list, scores_list, class_ids_list = postprocess(det_outputs)

            if isinstance(boxes_list, np.ndarray) or not isinstance(boxes_list, list):
                boxes_batch = [boxes_list]
                scores_batch = [scores_list]
                class_batch = [class_ids_list]
            else:
                boxes_batch = boxes_list
                scores_batch = scores_list
                class_batch = class_ids_list

            targets_np = targets.cpu().numpy()
            for local_idx in range(images.shape[0]):
                sample_id = sample_offset + local_idx
                gt_rows = targets_np[targets_np[:, 0] == local_idx]
                gt_boxes = xywh2xyxy(gt_rows[:, 2:6])
                gt_boxes[:, [0, 2]] *= postprocess.image_width
                gt_boxes[:, [1, 3]] *= postprocess.image_height
                gt_classes = gt_rows[:, 1].astype(int)

                pred_boxes = np.asarray(boxes_batch[local_idx]) if len(boxes_batch) > local_idx else np.zeros((0, 4))
                pred_scores = np.asarray(scores_batch[local_idx]) if len(scores_batch) > local_idx else np.zeros((0,))
                pred_classes = np.asarray(class_batch[local_idx]) if len(class_batch) > local_idx else np.zeros((0,), dtype=int)

                for cid in id_to_name:
                    per_class_gts[cid][sample_id] = gt_boxes[gt_classes == cid]

                for cid in id_to_name:
                    mask = pred_classes == cid
                    for box, score in zip(pred_boxes[mask], pred_scores[mask]):
                        per_class_preds[cid].append((sample_id, float(score), box))
            sample_offset += images.shape[0]

    ap_values = {}
    for cid in id_to_name:
        gt_entries = per_class_gts[cid]
        pred_entries = sorted(per_class_preds[cid], key=lambda x: x[1], reverse=True)
        npos = sum(len(boxes) for boxes in gt_entries.values())
        if npos == 0:
            continue

        gt_used = {sample_id: np.zeros(len(boxes), dtype=bool) for sample_id, boxes in gt_entries.items()}
        tp = np.zeros(len(pred_entries))
        fp = np.zeros(len(pred_entries))

        for i, (sample_id, score, pred_box) in enumerate(pred_entries):
            gt_boxes = gt_entries.get(sample_id, np.zeros((0, 4)))
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            ious = compute_iou(pred_box, gt_boxes)
            best = int(np.argmax(ious))
            if ious[best] >= 0.5 and not gt_used[sample_id][best]:
                tp[i] = 1
                gt_used[sample_id][best] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(npos, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap_values[cid] = _average_precision(recalls, precisions)

    map50 = float(np.mean(list(ap_values.values()))) if ap_values else 0.0
    return map50, ap_values


def save_checkpoint(path: str, model, optimizer=None, scaler=None, epoch: int = 0, metrics: Optional[dict] = None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
