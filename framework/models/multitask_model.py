import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import define_mobilenet
from .decoder import define_lightweight_refinenet, _make_crp


class HydraNet(nn.Module):
    def __init__(self, num_tasks=2, num_classes=6):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.detect_head = None
        self.only_return_object_detection_result = False

        # Bind _make_crp so it can be used in define_lightweight_refinenet
        self._make_crp = _make_crp.__get__(self)

        # Build encoder and decoder
        define_mobilenet(self)
        define_lightweight_refinenet(self)

    def attach_detection_head(self, detection_head):
        """Attach a YOLO head to the shared decoder features."""
        self.detect_head = detection_head
        return self

    def forward(self, x):
        # --- Encoder ---
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)

        # --- Decoder ---
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = F.interpolate(l7, size=l6.size()[2:], mode='bilinear', align_corners=False)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = F.interpolate(l5, size=l4.size()[2:], mode='bilinear', align_corners=False)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=False)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # --- Heads ---
        out_segm = self.relu(self.pre_segm(l3))
        out_segm = self.segm(out_segm)

        out_depth = self.relu(self.pre_depth(l3))
        out_depth = self.depth(out_depth)

        if self.num_tasks == 3:
            out_normal = self.relu(self.pre_normal(l3))
            out_normal = self.normal(out_normal)
            if self.detect_head is not None:
                out_det = self.detect_head([l3, l5, l7])
                if self.only_return_object_detection_result:
                    return out_det
                return out_segm, out_depth, out_normal, out_det
            return out_segm, out_depth, out_normal

        if self.detect_head is not None:
            out_det = self.detect_head([l3, l5, l7])
            if self.only_return_object_detection_result:
                return out_det
            return out_segm, out_depth, out_det

        return out_segm, out_depth
