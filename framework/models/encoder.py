import torch
import torch.nn as nn

# ---------- General utility layers ----------
def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=bias, groups=groups)

def batchnorm(num_features):
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)

def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=int(kernel_size / 2.), groups=groups, bias=False),
        batchnorm(out_channels)
    ]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)

# ---------- MobileNet encoder ----------
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(
            convbnrelu(in_planes, intermed_planes, 1),
            convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
            convbnrelu(intermed_planes, out_planes, 1, act=False)
        )

    def forward(self, x):
        out = self.output(x)
        return out + x if self.residual else out

def define_mobilenet(self):
    mobilenet_config = [[1, 16, 1, 1],
                        [6, 24, 2, 2],
                        [6, 32, 3, 2],
                        [6, 64, 4, 2],
                        [6, 96, 3, 1],
                        [6, 160, 3, 2],
                        [6, 320, 1, 1]]

    self.in_channels = 32
    self.num_layers = len(mobilenet_config)
    self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)

    c_layer = 2
    for t, c, n, s in mobilenet_config:
        layers = []
        for idx in range(n):
            layers.append(InvertedResidualBlock(
                self.in_channels, c, expansion_factor=t, stride=s if idx == 0 else 1))
            self.in_channels = c
        setattr(self, f'layer{c_layer}', nn.Sequential(*layers))
        c_layer += 1
