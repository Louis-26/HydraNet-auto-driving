import torch
import torch.nn as nn

from .detection_utils import dist2bbox, make_anchors, autopad


class Conv(nn.Module):
    """Standard convolution + BN + activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    """Integral module for Distribution Focal Loss."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DetectionHead(nn.Module):
    """
    YOLOv8-style head adapted to the shared HydraNet encoder/decoder features.

    The head consumes three feature maps from the decoder and returns:
    - training: raw feature tensors for loss computation
    - eval: decoded detections plus raw feature tensors
    """

    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(
        self,
        num_classes=80,
        decoder_channels=(256, 256, 256),
        head_channels=(64, 128, 256),
        stride=(8, 16, 32),
        reg_max=16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        self.reg_max = reg_max
        self.num_heads = len(head_channels)
        self.no = self.num_classes + self.reg_max * 4
        self.stride = torch.tensor(stride)

        self.p = nn.ModuleList(
            nn.Conv2d(in_, out_, 3, padding=1)
            for in_, out_ in zip(decoder_channels, head_channels)
        )
        self.pool = nn.MaxPool2d(2)

        c2 = max((16, head_channels[0] // 4, self.reg_max * 4))
        c3 = max(head_channels[0], self.num_classes)

        self.bbox_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in head_channels
        )
        self.class_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_classes, 1))
            for x in head_channels
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        features = list(x)
        for i in range(self.num_heads):
            xin = self.pool(self.p[i](features[i]))
            box_out = self.bbox_layers[i](xin)
            cls_out = self.class_layers[i](xin)
            features[i] = torch.cat((box_out, cls_out), 1)

        if self.training:
            return features

        shape = features[0].shape
        if self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(features, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in features], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y, features
