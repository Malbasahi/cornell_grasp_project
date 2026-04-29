"""
models/robust_model.py
-----------------------
Occlusion-robust grasp detector.

Extends the baseline in two ways:

1. INPUT CHANNEL EXTENSION
   Input = [depth, occlusion_map] — 2 channels instead of 1.
   The network sees where depth is missing/uncertain and can
   learn to down-weight unreliable regions.

2. OCCLUSION FEATURE MODULATION (optional, controlled by config)
   After the bottleneck, apply FiLM-style modulation:
       feat' = γ(occ_summary) * feat + β(occ_summary)
   where occ_summary = global average pooled occlusion map.
   This lets the network globally adapt its feature representation
   based on how occluded the whole scene is.

Everything else is identical to the baseline — same encoder,
decoder, loss, evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_model import (
    ConvBNReLU, EncoderBlock, DecoderBlock, GraspLoss
)


# ─────────────────────────────────────────────────────────────
# Occlusion feature modulation (FiLM)
# ─────────────────────────────────────────────────────────────

class OcclusionModulation(nn.Module):
    """
    FiLM-style feature modulation conditioned on occlusion severity.

    The occlusion map is globally average-pooled to a scalar,
    then passed through two small MLPs to produce per-channel
    scale (γ) and shift (β) vectors.

    feat' = (1 + γ) * feat + β       [residual γ for stability]

    Args:
        feature_channels (int): number of channels to modulate
    """

    def __init__(self, feature_channels):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(True),
            nn.Linear(32, feature_channels),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(True),
            nn.Linear(32, feature_channels),
        )

    def forward(self, feat, occ_map):
        """
        Args:
            feat    (Tensor): (B, C, H, W) feature map
            occ_map (Tensor): (B, 1, H, W) occlusion map

        Returns:
            modulated (Tensor): (B, C, H, W)
        """
        # Summarise occlusion map as a single scalar per image
        occ_scalar = occ_map.mean(dim=[2, 3])            # (B, 1)

        gamma = self.gamma_net(occ_scalar)               # (B, C)
        beta  = self.beta_net(occ_scalar)                # (B, C)

        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.view(*gamma.shape, 1, 1)
        beta  = beta.view(*beta.shape, 1, 1)

        return (1.0 + gamma) * feat + beta


# ─────────────────────────────────────────────────────────────
# Robust model
# ─────────────────────────────────────────────────────────────

class RobustGraspCNN(nn.Module):
    """
    Occlusion-robust encoder–decoder grasp detector.

    Identical to BaselineGraspCNN but:
        - Accepts 2-channel input (depth + occlusion map)
        - Optionally applies FiLM modulation at the bottleneck

    Args:
        input_channels      (int):  must be 2 for depth+occ
        encoder_channels    (list)
        decoder_channels    (list)
        dropout_rate        (float)
        use_occ_modulation  (bool): enable FiLM modulation
    """

    def __init__(self,
                 input_channels=2,
                 encoder_channels=(32, 64, 128, 256),
                 decoder_channels=(128, 64, 32),
                 dropout_rate=0.1,
                 use_occ_modulation=True):
        super().__init__()

        enc_ch = list(encoder_channels)
        dec_ch = list(decoder_channels)

        self.use_occ_modulation = use_occ_modulation

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

        # ── Occlusion modulation at bottleneck ────────────────
        if use_occ_modulation:
            self.occ_modulation = OcclusionModulation(enc_ch[-1])

        # ── Decoder ───────────────────────────────────────────
        self.dec_blocks = nn.ModuleList()
        d_in   = enc_ch[-1]
        skips  = list(reversed(enc_ch))

        for i, d_out in enumerate(dec_ch):
            skip_c = skips[i]
            self.dec_blocks.append(DecoderBlock(d_in, skip_c, d_out))
            d_in = d_out

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
            x (Tensor): (B, 2, H, W) — channel 0: depth, channel 1: occ map

        Returns:
            quality, angle, width — each (B, 1, H, W)
        """
        # Separate occlusion map for modulation
        occ_map = x[:, 1:2, :, :]   # (B, 1, H, W)

        # ── Encode ────────────────────────────────────────────
        skips = []
        for block in self.enc_blocks:
            x, skip = block(x)
            skips.append(skip)

        # ── Bottleneck ────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Occlusion modulation ──────────────────────────────
        if self.use_occ_modulation:
            # Downsample occ_map to match bottleneck spatial size
            occ_down = F.adaptive_avg_pool2d(occ_map, x.shape[-2:])
            x = self.occ_modulation(x, occ_down)

        # ── Decode ────────────────────────────────────────────
        rev_skips = list(reversed(skips))
        for i, block in enumerate(self.dec_blocks):
            x = block(x, rev_skips[i])

        for i, up_conv in enumerate(self.extra_up_convs):
            skip = rev_skips[len(self.dec_blocks) + i]
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up_conv[1](x)

        # ── Output heads ──────────────────────────────────────
        x = self.dropout(x)

        quality = torch.sigmoid(self.head_quality(x))
        angle   = torch.tanh(self.head_angle(x)) * (torch.pi / 2.0)
        width   = torch.sigmoid(self.head_width(x))

        return quality, angle, width

    @classmethod
    def from_config(cls, cfg):
        return cls(
            input_channels     = cfg['model']['input_channels'],
            encoder_channels   = cfg['model']['encoder_channels'],
            decoder_channels   = cfg['model']['decoder_channels'],
            dropout_rate       = cfg['model'].get('dropout_rate', 0.1),
            use_occ_modulation = cfg['model'].get('use_occ_modulation', True),
        )


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = RobustGraspCNN(input_channels=2)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")

    x = torch.randn(2, 2, 480, 640)
    q, a, w = model(x)
    print(f"quality: {q.shape}  angle: {a.shape}  width: {w.shape}")
    assert q.shape == (2, 1, 480, 640)
    print("Robust model OK")
