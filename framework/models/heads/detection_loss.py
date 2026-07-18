import torch
import torch.nn as nn

from .bbox_loss import BboxLoss
from .detection_utils import dist2bbox, make_anchors, xywh2xyxy
from .tal import TaskAlignedAssigner


class DetectionLoss(nn.Module):
    """YOLOv8 detection loss adapted for the HydraNet head."""

    def __init__(
        self,
        image_height,
        image_width,
        num_classes,
        stride,
        reg_max,
        box_loss_gain=7.5,
        cls_loss_gain=0.5,
        dfl_loss_gain=1.5,
        device="cpu",
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_loss_gain = box_loss_gain
        self.cls_loss_gain = cls_loss_gain
        self.dfl_loss_gain = dfl_loss_gain
        self.device = device

        self.no = self.num_classes + self.reg_max * 4
        self.stride = torch.tensor(stride)
        self.use_dfl = self.reg_max > 1

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.num_classes, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def encode_gt(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def decode_bbox(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds, targets):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        batch_size = feats[0].shape[0]

        pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.no, -1) for xi in feats], dim=2).split(
            (self.reg_max * 4, self.num_classes), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor([self.image_height, self.image_width], device=self.device, dtype=dtype)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = self.encode_gt(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.decode_bbox(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp_min(1.0)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.box_loss_gain
        loss[1] *= self.cls_loss_gain
        loss[2] *= self.dfl_loss_gain
        return loss.sum(), loss.detach()
