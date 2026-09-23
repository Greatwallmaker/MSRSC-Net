# Third-party sources

This release was organized from the authors' full MSRSC-Net experiment implementation. Module names have been aligned with the manuscript, the architecture has been fixed to the complete method, inactive ablation paths and single-frame video machinery have been removed, and the backbone has been made usable on CPU as well as CUDA.

## PVTv2

- Project: https://github.com/whai362/PVT
- Authors: Wenhai Wang and contributors.
- Work: *PVT v2: Improved Baselines with Pyramid Vision Transformer*.
- Source license: Apache License 2.0, reproduced in [licenses/PVT-Apache-2.0.txt](licenses/PVT-Apache-2.0.txt).
- Used in `msrsc_net/backbone.py`: PVTv2-B4 architecture, overlapping patch embeddings, spatial-reduction attention and depthwise-convolution MLPs. The efficient SDPA implementation also follows the implementation distributed with ZoomNeXt.

## ZoomNeXt

- Project: https://github.com/lartpang/ZoomNeXt
- Authors: Youwei Pang and contributors.
- Work: *ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection*.
- Used in `msrsc_net/modules.py`, `model.py` and `loss.py`: three-scale framework, MHSIU, RGPU, supporting projection utilities and uncertainty-aware loss.
- No standalone license file was present in the supplied source package or the upstream repository root when this release was prepared. This repository does not assign a new license to those upstream components.

SOCA, RGMF, SCEM, their integration and the full training objective are retained from the authors' supplied MSRSC-Net implementation. Source attribution remains attached to the adapted components.
