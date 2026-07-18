import torch.nn as nn
from .encoder import conv1x1, conv3x3  # Reuse your helpers

class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__()
        for i in range(n_stages):
            setattr(self, f'{i + 1}_outvar_dimred',
                    conv1x1(in_planes if i == 0 else out_planes,
                            out_planes, stride=1, bias=False,
                            groups=in_planes if groups else 1))
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, f'{i + 1}_outvar_dimred')(top)
            x = top + x
        return x

def _make_crp(self, in_planes, out_planes, stages, groups=False):
    return nn.Sequential(CRPBlock(in_planes, out_planes, stages, groups=groups))

def define_lightweight_refinenet(self):
    self.conv8 = conv1x1(320, 256, bias=False)
    self.conv7 = conv1x1(160, 256, bias=False)
    self.conv6 = conv1x1(96, 256, bias=False)
    self.conv5 = conv1x1(64, 256, bias=False)
    self.conv4 = conv1x1(32, 256, bias=False)
    self.conv3 = conv1x1(24, 256, bias=False)

    self.crp4 = self._make_crp(256, 256, 4)
    self.crp3 = self._make_crp(256, 256, 4)
    self.crp2 = self._make_crp(256, 256, 4)
    self.crp1 = self._make_crp(256, 256, 4, groups=True)

    self.conv_adapt4 = conv1x1(256, 256, bias=False)
    self.conv_adapt3 = conv1x1(256, 256, bias=False)
    self.conv_adapt2 = conv1x1(256, 256, bias=False)

    self.pre_depth = conv1x1(256, 256, groups=256, bias=False)
    self.depth = conv3x3(256, 1, bias=True)

    self.pre_segm = conv1x1(256, 256, groups=256, bias=False)
    self.segm = conv3x3(256, self.num_classes, bias=True)

    self.relu = nn.ReLU6(inplace=True)

    if self.num_tasks == 3:
        self.pre_normal = conv1x1(256, 256, groups=256, bias=False)
        self.normal = conv3x3(256, 3, bias=True)
