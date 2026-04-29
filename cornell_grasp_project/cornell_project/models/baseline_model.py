"""
models/baseline_model.py
-------------------------
Baseline CNN grasp detector.

Architecture: Encoder–decoder (U-Net style) with three output heads.

Input : (B, 1, H, W) — single-channel normalised depth image
Output: three pixel maps, each (B, 1, H, W):
    quality : predicted grasp quality in [0, 1]  (sigmoid)
    angle   : predicted grasp angle in (-π/2, π/2)  (tanh × π/2)
    width   : predicted normalised gripper width in [0, 1]  (sigmoid)

The encoder progressively downsamples with strided convolutions.
The decoder uses bilinear upsampling + skip connections from encoder.
Three separate 1×1 conv heads produce the final maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Two ConvBNReLU + strided downsample."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch,  out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.down  = nn.Conv2d(out_ch, out_ch, 3, stride=2,
                               padding=1, bias=False)

    def forward(self, x):
        x   = self.conv1(x)
        x   = self.conv2(x)
        skip = x                        # save for skip connection
        x   = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
    """Bilinear upsample + skip concat + two ConvBNReLU."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch,          out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ─────────────────────────────────────────────────────────────
# Baseline model
# ─────────────────────────────────────────────────────────────

class BaselineGraspCNN(nn.Module):
    """
    Encoder–decoder CNN for pixel-wise grasp prediction.

    Args:
        input_channels   (int): 1 (depth) or 2 (depth + occ map)
        encoder_channels (list): feature channels per encoder stage
        decoder_channels (list): feature channels per decoder stage
        dropout_rate     (float): dropout before output heads
    """

    def __init__(self,
                 input_channels=1,
                 encoder_channels=(32, 64, 128, 256),
                 decoder_channels=(128, 64, 32),
                 dropout_rate=0.1):
        super().__init__()

        enc_ch = list(encoder_channels)
        dec_ch = list(decoder_channels)

        # ── Encoder ───────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        in_ch = input_channels
        for out_ch in enc_ch:
            self.enc_blocks.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBNReLU(enc_ch[-1], enc_ch[-1] * 2),
            ConvBNReLU(enc_ch[-1] * 2, enc_ch[-1]),
        )

        # ── Decoder ───────────────────────────────────────────
        # decoder input = bottleneck ch; skip from matching encoder block
        self.dec_blocks = nn.ModuleList()
        d_in   = enc_ch[-1]
        skips  = list(reversed(enc_ch))   # skip channels

        for i, d_out in enumerate(dec_ch):
            skip_c = skips[i]
            self.dec_blocks.append(DecoderBlock(d_in, skip_c, d_out))
            d_in = d_out

        # Final upsample to restore full resolution if more enc than dec stages
        self.extra_up_convs = nn.ModuleList()
        remaining_skips = skips[len(dec_ch):]
        for sk in remaining_skips:
            self.extra_up_convs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False),
                    ConvBNReLU(d_in + sk, d_in),
                )
            )

        # ── Output heads ──────────────────────────────────────
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.head_quality = nn.Conv2d(d_in, 1, kernel_size=1)
        self.head_angle   = nn.Conv2d(d_in, 1, kernel_size=1)
        self.head_width   = nn.Conv2d(d_in, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, H, W)

        Returns:
            quality (Tensor): (B, 1, H, W) in [0,1]
            angle   (Tensor): (B, 1, H, W) in (-π/2, π/2)
            width   (Tensor): (B, 1, H, W) in [0,1]
        """
        # ── Encode ────────────────────────────────────────────
        skips = []
        for block in self.enc_blocks:
            x, skip = block(x)
            skips.append(skip)

        # ── Bottleneck ────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decode ────────────────────────────────────────────
        rev_skips = list(reversed(skips))
        for i, block in enumerate(self.dec_blocks):
            x = block(x, rev_skips[i])

        # Extra upsamples if encoder has more stages than decoder
        for i, up_conv in enumerate(self.extra_up_convs):
            skip = rev_skips[len(self.dec_blocks) + i]
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up_conv[1](x)   # ConvBNReLU only

        # ── Output heads ──────────────────────────────────────
        x = self.dropout(x)

        quality = torch.sigmoid(self.head_quality(x))
        # tanh output → scale to (-π/2, π/2)
        angle   = torch.tanh(self.head_angle(x)) * (torch.pi / 2.0)
        width   = torch.sigmoid(self.head_width(x))

        return quality, angle, width

    @classmethod
    def from_config(cls, cfg):
        return cls(
            input_channels   = cfg['model']['input_channels'],
            encoder_channels = cfg['model']['encoder_channels'],
            decoder_channels = cfg['model']['decoder_channels'],
            dropout_rate     = cfg['model'].get('dropout_rate', 0.1),
        )


# ─────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────

class GraspLoss(nn.Module):
    """
    Combined loss for the three prediction heads.

    quality : Binary Cross-Entropy (pixel-wise)
    angle   : Cosine loss — penalises angular error symmetrically
    width   : Smooth L1 (Huber) — only at valid grasp pixels

    Args:
        quality_weight (float)
        angle_weight   (float)
        width_weight   (float)
    """

    def __init__(self, quality_weight=1.0, angle_weight=1.0,
                 width_weight=0.5):
        super().__init__()
        self.w_q = quality_weight
        self.w_a = angle_weight
        self.w_w = width_weight
        self.bce   = nn.BCELoss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, pred_q, pred_a, pred_w,
                gt_q,   gt_a,   gt_w):
        """
        Args:
            pred_q, pred_a, pred_w : (B, 1, H, W) predictions
            gt_q,   gt_a,   gt_w   : (B, 1, H, W) ground truth

        Returns:
            total_loss (Tensor): scalar
            loss_dict  (dict):   individual losses for logging
        """
        # Quality loss — all pixels
        q_loss = self.bce(pred_q, gt_q)

        # Angle loss — only at valid grasp pixels (gt_quality == 1)
        # Use cosine similarity: 1 - cos(2*(pred - gt)) to handle ±π/2 symmetry
        mask   = (gt_q > 0.5).squeeze(1)   # (B, H, W) bool
        if mask.sum() > 0:
            pa = pred_a.squeeze(1)[mask]
            ga = gt_a.squeeze(1)[mask]
            a_loss = 1.0 - torch.cos(2.0 * (pa - ga))
            a_loss = a_loss.mean()
        else:
            a_loss = torch.tensor(0.0, device=pred_q.device)

        # Width loss — only at valid grasp pixels
        if mask.sum() > 0:
            pw = pred_w.squeeze(1)[mask]
            gw = gt_w.squeeze(1)[mask]
            w_loss = self.huber(pw, gw)
        else:
            w_loss = torch.tensor(0.0, device=pred_q.device)

        total = self.w_q * q_loss + self.w_a * a_loss + self.w_w * w_loss

        return total, {
            'quality_loss': q_loss.item(),
            'angle_loss':   a_loss.item(),
            'width_loss':   w_loss.item(),
            'total_loss':   total.item(),
        }


# ─────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = BaselineGraspCNN(input_channels=1)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")

    x   = torch.randn(2, 1, 480, 640)
    q, a, w = model(x)
    print(f"quality: {q.shape}  angle: {a.shape}  width: {w.shape}")
    assert q.shape == (2, 1, 480, 640)
    assert q.min() >= 0 and q.max() <= 1

    loss_fn = GraspLoss()
    gt_q = (torch.rand(2, 1, 480, 640) > 0.9).float()
    gt_a = torch.zeros(2, 1, 480, 640)
    gt_w = torch.zeros(2, 1, 480, 640)
    loss, d = loss_fn(q, a, w, gt_q, gt_a, gt_w)
    print(f"Loss: {loss.item():.4f}  dict: {d}")
    print("Baseline model OK")
