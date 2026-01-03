import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom


# =========================
# Basic Blocks
# =========================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvWithAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 8, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        att = self.att(feat)
        return feat * att


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvWithAttention(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConvWithAttention(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1, dilation=1, padding=0),
            nn.Conv2d(in_ch, out_ch, 3, dilation=6, padding=6),
            nn.Conv2d(in_ch, out_ch, 3, dilation=12, padding=12),
            nn.Conv2d(in_ch, out_ch, 3, dilation=18, padding=18),
        ])
        self.project = nn.Conv2d(out_ch * 4, out_ch, 1)

    def forward(self, x):
        feats = [block(x) for block in self.blocks]
        x = torch.cat(feats, dim=1)
        return self.project(x)


# =========================
# UNetV2 (核心模型)
# =========================

class UNetV2(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_ch=64):
        super().__init__()

        self.inc = DoubleConvWithAttention(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.aspp = ASPP(base_ch * 16, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)

        self.outc = nn.Sequential(
            ConvBNReLU(base_ch, base_ch // 2),
            nn.Conv2d(base_ch // 2, out_channels, 1)
        )

        self._init_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.aspp(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# =========================
# Sequence Inference
# =========================

class SequenceInference:
    def __init__(self, model):
        self.model = model

    def infer(self, frames, init_mask):
        preds = []
        prev_mask = init_mask

        for t in range(frames.shape[0]):
            inp = np.concatenate([frames[t:t+1], prev_mask], axis=1)
            inp = torch.from_numpy(inp).float().cuda()

            with torch.no_grad():
                pred = torch.sigmoid(self.model(inp))[0, 0].cpu().numpy()

            pred = self._post_process(pred, prev_mask[0, 0])
            preds.append(pred)
            prev_mask = pred[None, None]

        return np.stack(preds)

    def _post_process(self, pred, prev_mask):
        pred = zoom(pred, prev_mask.shape[0] / pred.shape[0], order=1)
        return pred > 0.5


# =========================
# Metrics (可选但推荐)
# =========================

def dice_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return 2 * inter / (pred.sum() + target.sum() + 1e-6)


def center_distance(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return np.inf
    py, px = np.argwhere(pred).mean(axis=0)
    ty, tx = np.argwhere(target).mean(axis=0)
    return np.sqrt((py - ty) ** 2 + (px - tx) ** 2)
