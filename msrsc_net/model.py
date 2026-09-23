"""MSRSC-Net: the fixed, complete three-scale model used in the paper."""

from torch import nn

from .backbone import PVTv2B4
from .modules import (
    MHSIU,
    RGMF,
    RGPU,
    SCEM,
    SOCA,
    BoundaryRefiner,
    ConvBNReLU,
    PixelNormalizer,
    RegionPrior,
    resize_to,
)


class MSRSCNet(nn.Module):
    """DOM-only binary segmentation. Input: float RGB [N, 3, H, W] in [0, 1].

    Output: raw foreground logits [N, 1, H, W]. ImageNet normalization is
    internal. The paper uses H=W=384 and scales (1.5, 1.0, 0.5).
    """

    def __init__(self):
        super().__init__()
        self.encoder = PVTv2B4()
        self.projections = nn.ModuleDict()
        self.fusions = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.scem = nn.ModuleDict()
        for level, channels in ((5, 512), (4, 320), (3, 128), (2, 64)):
            key = str(level)
            self.projections[key] = (
                SOCA(channels, 64, is_top=level == 5)
                if level >= 4
                else ConvBNReLU(channels, 64, 3, 1, 1)
            )
            self.fusions[key] = RGMF(64) if level >= 4 else MHSIU(64)
            self.decoders[key] = RGPU(64)
            self.scem[key] = SCEM(64)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 64, 3, 1, 1),
        )
        self.region_prior = RegionPrior(64)
        self.boundary_refiner = BoundaryRefiner(64)
        self.normalizer = PixelNormalizer()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    @staticmethod
    def _validate_image(image):
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Expected a floating-point RGB tensor [N, 3, H, W].")
        h, w = image.shape[-2:]
        if min(h, w) < 64 or h % 32 or w % 32:
            raise ValueError(
                "Reference height and width must be multiples of 32 and at least 64."
            )

    def forward(self, image):
        self._validate_image(image)
        h, w = image.shape[-2:]
        return self.forward_scales(
            resize_to(image, (h * 3 // 2, w * 3 // 2)),
            image,
            resize_to(image, (h // 2, w // 2)),
        )

    def forward_scales(self, image_l, image_m, image_s):
        """Use externally prepared RGB scales, e.g. the original experiment loader.

        l/m/s denote enlarged/reference/reduced resolution. Each tensor must
        be unnormalized [0, 1] RGB; branch sizes must be 1.5/1.0/0.5 times m.
        """
        self._validate_image(image_m)
        h, w = image_m.shape[-2:]
        for image, size in (
            (image_l, (h * 3 // 2, w * 3 // 2)),
            (image_s, (h // 2, w // 2)),
        ):
            if image.shape != (image_m.shape[0], 3, *size):
                raise ValueError(
                    "Branch shapes must match the batch and scale factors 1.5/1.0/0.5."
                )
            if image.device != image_m.device or image.dtype != image_m.dtype:
                raise ValueError(
                    "All three branches must share the same device and dtype."
                )
        features = [
            self.encoder(self.normalizer(x)) for x in (image_l, image_m, image_s)
        ]
        x = None
        for level in (5, 4, 3, 2):
            key = str(level)
            l, m, s = [self.projections[key](f[f"reduction_{level}"]) for f in features]
            fused = self.fusions[key](l, m, s)
            if x is not None:
                fused = fused + resize_to(x, fused.shape[-2:])
            x = self.scem[key](self.decoders[key](fused))
        x = self.region_prior(self.upsample(x))
        logits = self.predictor(x)
        return logits + self.boundary_refiner(x, logits.shape[-2:])
