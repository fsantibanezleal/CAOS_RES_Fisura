"""HrSegNet, reimplemented faithfully in PyTorch from the published description (ladder B).

Source design: Li, Ma, Liu, Cheng, "Real-time high-resolution neural network with semantic
guidance for crack segmentation", Automation in Construction 156:105112 (2023); reference
implementation CHDyshli/HrSegNet4CrackSegmentation (Apache-2.0, PaddlePaddle). This module is an
in-repo PyTorch re-expression of that architecture per the research dossier:

- Stem to 1/4 resolution, then a HIGH-RESOLUTION path that stays at 1/4 the whole way: three
  HrSeg blocks, each three Conv-BN-ReLU layers (stride 1), constant base width B (16/32/48).
- An auxiliary SEMANTIC-GUIDANCE path: progressively downsampled context features, fused into the
  high-resolution path at each stage by elementwise summation + ReLU (guidance, not concat).
- Head: 3x3 transposed convolution (stride 2) to 1/2 resolution, then bilinear upsample to full;
  two output channels (background, crack).
- Loss (training module): primary cross-entropy + 0.5 x two auxiliary cross-entropies on
  intermediate heads, per the published recipe.

Torch imported lazily (GPU lane only).
"""
from __future__ import annotations


def build_hrsegnet(base: int = 16, num_classes: int = 2):
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class ConvBNReLU(nn.Sequential):
        def __init__(self, cin, cout, k=3, s=1):
            super().__init__(
                nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

    class HrSegBlock(nn.Module):
        """Three Conv-BN-ReLU layers at constant resolution and width."""

        def __init__(self, ch):
            super().__init__()
            self.body = nn.Sequential(ConvBNReLU(ch, ch), ConvBNReLU(ch, ch), ConvBNReLU(ch, ch))

        def forward(self, x):
            return self.body(x)

    class GuidanceStage(nn.Module):
        """Downsample the guidance stream, project to the high-res width, upsample, fuse by sum+ReLU."""

        def __init__(self, g_in, g_out, hr_ch):
            super().__init__()
            self.down = ConvBNReLU(g_in, g_out, k=3, s=2)
            self.proj = nn.Conv2d(g_out, hr_ch, 1, bias=False)
            self.bn = nn.BatchNorm2d(hr_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, hr):
            g = self.down(g)
            up = nn.functional.interpolate(self.bn(self.proj(g)), size=hr.shape[-2:], mode="bilinear", align_corners=False)
            return g, self.relu(hr + up)

    class AuxHead(nn.Module):
        def __init__(self, ch, num_classes):
            super().__init__()
            self.head = nn.Conv2d(ch, num_classes, 1)

        def forward(self, x, size):
            return nn.functional.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)

    class HrSegNet(nn.Module):
        def __init__(self, base=16, num_classes=2):
            super().__init__()
            self.stem = nn.Sequential(ConvBNReLU(3, base, s=2), ConvBNReLU(base, base, s=2))  # 1/4 res
            self.hr1, self.hr2, self.hr3 = HrSegBlock(base), HrSegBlock(base), HrSegBlock(base)
            self.g1 = GuidanceStage(base, base * 2, base)      # guidance at 1/8
            self.g2 = GuidanceStage(base * 2, base * 4, base)  # guidance at 1/16
            self.aux1, self.aux2 = AuxHead(base, num_classes), AuxHead(base, num_classes)
            self.head = nn.Sequential(
                nn.ConvTranspose2d(base, base, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
                nn.Conv2d(base, num_classes, 1),
            )

        def forward(self, x):
            size = x.shape[-2:]
            h = self.stem(x)
            g = h
            h = self.hr1(h)
            g, h = self.g1(g, h)
            a1 = self.aux1(h, size)
            h = self.hr2(h)
            g, h = self.g2(g, h)
            a2 = self.aux2(h, size)
            h = self.hr3(h)
            out = nn.functional.interpolate(self.head(h), size=size, mode="bilinear", align_corners=False)
            if self.training:
                return out, a1, a2
            return out

    torch.manual_seed(0)  # deterministic init layout; the trainer reseeds properly
    return HrSegNet(base=base, num_classes=num_classes)
