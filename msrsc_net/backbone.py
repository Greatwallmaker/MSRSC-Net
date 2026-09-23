"""Fixed PVTv2-B4 encoder for MSRSC-Net.

Adapted from PVT (Wenhai Wang and contributors) and the efficient PVTv2
implementation distributed with ZoomNeXt. Modified to retain B4 only, use
PyTorch SDPA on CPU or CUDA, and omit classification and variant switches.
See README.md acknowledgements and licenses/PVT-Apache-2.0.txt.
"""

import math

from torch import nn
from torch.nn import functional as F


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        fan_out = math.prod(module.kernel_size) * module.out_channels // module.groups
        nn.init.normal_(module.weight, std=math.sqrt(2 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = self.dwconv(x.transpose(1, 2).reshape(b, c, h, w))
        return x.flatten(2).transpose(1, 2)


class _MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = _DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, h, w):
        return self.fc2(self.act(self.dwconv(self.fc1(x), h, w)))


class _Attention(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.num_heads = heads
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = (
            self.q(x)
            .reshape(b, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        if self.sr_ratio > 1:
            reduced = self.sr(x.permute(0, 2, 1).reshape(b, c, h, w))
            x = self.norm(reduced.flatten(2).transpose(1, 2))
        kv = (
            self.kv(x)
            .reshape(b, -1, 2, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        x = F.scaled_dot_product_attention(q, kv[0], kv[1], dropout_p=0)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MLP(dim, int(dim * mlp_ratio))

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        return x + self.mlp(self.norm2(x), h, w)


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, dim, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size, stride, kernel_size // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class PVTv2B4(nn.Module):
    """Four-stage shared encoder with channels (64, 128, 320, 512)."""

    def __init__(self):
        super().__init__()
        self.embed_dims = (64, 128, 320, 512)
        heads, ratios, depths, reductions = (
            (1, 2, 5, 8),
            (8, 8, 4, 4),
            (3, 8, 27, 3),
            (8, 4, 2, 1),
        )
        for i, dim in enumerate(self.embed_dims):
            setattr(
                self,
                f"patch_embed{i + 1}",
                _OverlapPatchEmbed(
                    3 if i == 0 else self.embed_dims[i - 1],
                    dim,
                    7 if i == 0 else 3,
                    4 if i == 0 else 2,
                ),
            )
            setattr(
                self,
                f"block{i + 1}",
                nn.ModuleList(
                    [
                        _Block(dim, heads[i], ratios[i], reductions[i])
                        for _ in range(depths[i])
                    ]
                ),
            )
            setattr(self, f"norm{i + 1}", nn.LayerNorm(dim, eps=1e-6))
        self.apply(_init_weights)

    def forward(self, x):
        batch = x.shape[0]
        features = {}
        for i in range(1, 5):
            x, h, w = getattr(self, f"patch_embed{i}")(x)
            for block in getattr(self, f"block{i}"):
                x = block(x, h, w)
            x = getattr(self, f"norm{i}")(x)
            x = x.reshape(batch, h, w, -1).permute(0, 3, 1, 2).contiguous()
            features[f"reduction_{i + 1}"] = x
        return features
