"""Full MSRSC-Net modules. Adapted from the authors' experiment implementation.

MHSIU, RGPU and basic projection utilities derive from ZoomNeXt (Youwei Pang
and contributors). Modified for single-image MSRSC-Net; see THIRD_PARTY.md.
"""

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair as to_2tuple


def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    if act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if act_name == "gelu":
        return nn.GELU()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise NotImplementedError(act_name)


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
    ):
        super().__init__()
        conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class PixelNormalizer(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.register_buffer(name="mean", tensor=torch.Tensor(mean).reshape(3, 1, 1))
        self.register_buffer(name="std", tensor=torch.Tensor(std).reshape(3, 1, 1))

    def forward(self, x):
        return x.sub(self.mean).div(self.std)


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation
        )
        self.conv3x3_2 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation
        )
        self.conv3x3_3 = ConvBNReLU(
            out_dim, out_dim, 3, dilation=dilation, padding=dilation
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = resize_to(self.conv1x1_2(y0), tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        act_name="relu",
    ):
        super().__init__(
            ConvBNReLU(
                in_dim,
                in_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_dim,
            ),
            ConvBNReLU(in_dim, out_dim, 1, act_name=act_name),
        )


class RegionPrior(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.region_proj = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.texture_proj = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.region_gate = nn.Sequential(
            ConvBNReLU(2 * in_dim, in_dim, 1),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        smooth = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        texture = (x - smooth).abs()
        gate = self.region_gate(torch.cat([smooth, texture], dim=1))
        residual = gate * self.region_proj(smooth) + 0.5 * (
            1 - gate
        ) * self.texture_proj(texture)
        return x + self.alpha * residual


class BoundaryRefiner(nn.Module):
    def __init__(self, in_dim, mid_dim=32):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.refine = nn.Sequential(
            ConvBNReLU(in_dim, mid_dim, 3, 1, 1),
            ConvBNReLU(mid_dim, mid_dim, 3, 1, 1),
            nn.Conv2d(mid_dim, 1, 1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x, tgt_hw):
        return self.alpha * resize_to(self.refine(x), tgt_hw=tgt_hw)


class RGPU(nn.Module):
    """Recursive gated pyramid unit inherited from ZoomNeXt, for single images."""

    def __init__(self, in_c, num_groups=6, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(
                2 * hidden_dim, 3 * hidden_dim, 3, 1, 1
            )
        self.interact[str(num_groups - 1)] = ConvBNReLU(
            2 * hidden_dim, 2 * hidden_dim, 3, 1, 1
        )
        self.fuse = ConvBNReLU(num_groups * hidden_dim, in_c, 3, 1, 1, act_name=None)
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        outs, gates = ([], [])
        curr_x = xs[0]
        branch_out = self.interact["0"](curr_x)
        curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)
        for group_id in range(1, self.num_groups - 1):
            curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
            branch_out = self.interact[str(group_id)](curr_x)
            curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)
        curr_x = torch.cat([xs[self.num_groups - 1], curr_fork], dim=1)
        branch_out = self.interact[str(self.num_groups - 1)](curr_x)
        curr_out, curr_gate = branch_out.chunk(2, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)
        out = torch.cat(outs, dim=1)
        gate = self.gate_generator(torch.cat(gates, dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


class MHSIU(nn.Module):
    """Multi-scale hierarchical semantic interaction unit inherited from ZoomNeXt."""

    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_lms = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)
        self.initial_merge = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)
        self.num_groups = num_groups
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, l, m, s):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)
        l, m, s = (self.conv_l(l), self.conv_m(m), self.conv_s(s))
        lms = torch.cat([l, m, s], dim=1)
        attn = self.conv_lms(lms)
        attn = rearrange(
            attn, "bt (nb ng d) h w -> (bt ng) (nb d) h w", nb=3, ng=self.num_groups
        )
        attn = self.trans(attn).unsqueeze(dim=2)
        x = self.initial_merge(lms)
        x = rearrange(
            x, "bt (nb ng d) h w -> (bt ng) nb d h w", nb=3, ng=self.num_groups
        )
        x = (attn * x).sum(dim=1)
        return rearrange(x, "(bt ng) d h w -> bt (ng d) h w", ng=self.num_groups)


class SOCA(nn.Module):
    """Strip-Oriented Context Aggregation at L4 and L5; all three components enabled."""

    def __init__(self, in_dim, out_dim, is_top=False, kernel_size=11):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.02))
        self.base = (
            SimpleASPP(in_dim=in_dim, out_dim=out_dim)
            if is_top
            else ConvBNReLU(in_dim, out_dim, 3, 1, 1)
        )
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        branches = [
            DepthwiseSeparableConv(out_dim, out_dim, 3, padding=1),
            DepthwiseSeparableConv(out_dim, out_dim, 3, padding=2, dilation=2),
        ]
        branches.extend(
            [
                DepthwiseSeparableConv(
                    out_dim, out_dim, (1, kernel_size), padding=(0, pad)
                ),
                DepthwiseSeparableConv(
                    out_dim, out_dim, (kernel_size, 1), padding=(pad, 0)
                ),
            ]
        )
        self.branches = nn.ModuleList(branches)
        self.texture_gate = nn.Sequential(
            ConvBNReLU(2 * out_dim, out_dim, 1),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid(),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), ConvBNReLU(out_dim, out_dim, 1)
        )
        branch_count = 5
        self.fuse = nn.Sequential(
            ConvBNReLU(branch_count * out_dim, out_dim, 1),
            ConvBNReLU(out_dim, out_dim, 3, 1, 1, act_name=None),
        )
        nn.init.zeros_(self.fuse[-1][0].weight)

    def forward(self, x):
        base = self.base(x)
        x = base
        smooth = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        detail = (x - smooth).abs()
        gate = self.texture_gate(torch.cat([smooth, detail], dim=1))
        x = gate * x + (1 - gate) * smooth
        outs = [branch(x) for branch in self.branches]
        outs.append(resize_to(self.global_context(x), tgt_hw=x.shape[-2:]))
        return base + self.alpha * self.fuse(torch.cat(outs, dim=1))


class RGMF(nn.Module):
    """Reliability-Guided Multi-Scale Fusion at L4 and L5."""

    def __init__(self, in_dim, num_groups=4, ribbon_kernel_size=15):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.base_merge = MHSIU(in_dim, num_groups)
        self.alpha = nn.Parameter(torch.tensor(0.02))
        context_dim = 5 * in_dim
        self.scale_logits = nn.Sequential(
            ConvBNReLU(context_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )
        self.context_proj = ConvBNReLU(context_dim, in_dim, 1)
        self.reliability_gate = nn.Sequential(
            ConvBNReLU(2 * in_dim, in_dim, 1), nn.Conv2d(in_dim, 3, 1), nn.Sigmoid()
        )
        self.out_proj = ConvBNReLU(in_dim, in_dim, 3, 1, 1, act_name=None)
        nn.init.zeros_(self.out_proj[0].weight)
        kernel_size = int(ribbon_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        self.ribbon_context = nn.Sequential(
            DepthwiseSeparableConv(in_dim, in_dim, (1, kernel_size), padding=(0, pad)),
            DepthwiseSeparableConv(in_dim, in_dim, (kernel_size, 1), padding=(pad, 0)),
            DepthwiseSeparableConv(in_dim, in_dim, 3, padding=2, dilation=2),
        )
        self.ribbon_gate = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.Sigmoid(),
        )

    def _align_scales(self, l, m, s):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)
        return (self.conv_l(l), self.conv_m(m), self.conv_s(s))

    def forward(self, l, m, s):
        base = self.base_merge(l=l, m=m, s=s)
        l, m, s = self._align_scales(l=l, m=m, s=s)
        diff_lm = (l - m).abs()
        diff_sm = (s - m).abs()
        context = torch.cat([l, m, s, diff_lm, diff_sm], dim=1)
        logits = self.scale_logits(context)
        reliability = self.reliability_gate(torch.cat([diff_lm, diff_sm], dim=1))
        logits = logits + torch.log(reliability.clamp_min(0.0001))
        weights = logits.softmax(dim=1).unsqueeze(2)
        scales = torch.stack([l, m, s], dim=1)
        fused = (weights * scales).sum(dim=1)
        fused = fused + self.context_proj(context)
        detail = self.ribbon_context(fused)
        smooth = F.avg_pool2d(fused, kernel_size=5, stride=1, padding=2)
        gate = self.ribbon_gate(torch.cat([smooth, diff_lm + diff_sm, detail], dim=1))
        fused = fused + gate * detail
        return base + self.alpha * self.out_proj(fused)


class _ContinuityUnit(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim // 2, 16)
        self.pre = ConvBNReLU(in_dim, hidden_dim, 1)
        self.local_branch = DepthwiseSeparableConv(hidden_dim, hidden_dim, 3, padding=1)
        self.long_h_branch = DepthwiseSeparableConv(
            hidden_dim, hidden_dim, (1, 17), padding=(0, 8)
        )
        self.long_v_branch = DepthwiseSeparableConv(
            hidden_dim, hidden_dim, (17, 1), padding=(8, 0)
        )
        self.dilate_3_branch = DepthwiseSeparableConv(
            hidden_dim, hidden_dim, 3, padding=3, dilation=3
        )
        self.dilate_5_branch = DepthwiseSeparableConv(
            hidden_dim, hidden_dim, 3, padding=5, dilation=5
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.ReLU(True),
        )
        branch_count = 6
        self.branch_gate = nn.Sequential(
            ConvBNReLU(branch_count * hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, branch_count, 1),
            nn.Softmax(dim=1),
        )
        self.fuse = ConvBNReLU(hidden_dim, in_dim, 3, 1, 1, act_name=None)
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        feat = self.pre(x)
        branches = [
            self.local_branch(feat),
            self.long_h_branch(feat),
            self.long_v_branch(feat),
            self.dilate_3_branch(feat),
            self.dilate_5_branch(feat),
            resize_to(self.global_branch(feat), tgt_hw=feat.shape[-2:]),
        ]
        gate = self.branch_gate(torch.cat(branches, dim=1)).unsqueeze(2)
        residual = (gate * torch.stack(branches, dim=1)).sum(dim=1)
        return self.final_relu(self.fuse(residual) + x)


class SCEM(nn.Module):
    """Structural Continuity Enhancement after each L2--L5 RGPU decoder."""

    def __init__(self, in_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.continuity = _ContinuityUnit(in_dim)
        self.out_proj = ConvBNReLU(in_dim, in_dim, 3, 1, 1, act_name=None)
        nn.init.zeros_(self.out_proj[0].weight)

    def forward(self, x):
        residual = self.out_proj(self.continuity(x))
        return x + (self.alpha + 0.02) * residual
