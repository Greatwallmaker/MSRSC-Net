# MSRSC-Net

**Multi-Scale Reliability and Structural Continuity Network** for binary semantic segmentation of Great Wall remains from UAV digital orthophoto maps (DOMs).

This repository contains the complete model and training objective used in our study. The implementation uses RGB DOM input only and keeps a fixed architecture without ablation switches. Datasets, model weights, experiment logs, baseline models and evaluation scripts are not included.

## Architecture

A shared **PVTv2-B4** encoder processes enlarged, reference and reduced inputs. For a reference input of `384 × 384`, the branch sizes are `576 × 576`, `384 × 384` and `192 × 192`, respectively. The decoder width is 64 channels.

| Paper module | Code | Placement |
| --- | --- | --- |
| Strip-Oriented Context Aggregation (SOCA) | `SOCA` | L4 and L5 projections |
| Reliability-Guided Multi-Scale Fusion (RGMF) | `RGMF` | L4 and L5 fusion |
| Structural Continuity Enhancement (SCEM) | `SCEM` | After RGPU at L2–L5 |
| Multi-scale hierarchical semantic interaction unit | `MHSIU` | L2 and L3 fusion; also the base fusion path inside RGMF |
| Recursive gated pyramid unit | `RGPU` | L2–L5 decoding |

SOCA retains texture gating, local/dilated convolutions, horizontal/vertical strip branches and global context. RGMF retains cross-scale contrast, reliability weighting and strip-context refinement. The output head retains region-prior and boundary-residual refinement. MHSIU and RGPU are inherited from ZoomNeXt; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Files

```text
msrsc_net/
  backbone.py   # shared PVTv2-B4 encoder
  modules.py    # SOCA, RGMF, SCEM and supporting layers
  model.py      # complete MSRSCNet architecture
  loss.py       # complete MSRSCLoss objective
  weights.py    # strict loading and original checkpoint name mapping
  __init__.py   # public imports
```

## Installation

Use Python 3.10 or later. Install a PyTorch build suitable for your CPU/CUDA environment, then install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The model requires only PyTorch and einops. It does not download weights automatically.

## Inference

```python
import torch
from msrsc_net import MSRSCNet, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSRSCNet()
# Required for meaningful predictions; weights are not bundled:
# load_checkpoint(model, 'path/to/trained_weights.pth')
model = model.to(device).eval()

# Example input only: replace with your RGB DOM tensor in [0, 1].
image = torch.rand(1, 3, 384, 384, device=device)
with torch.inference_mode():
    logits = model(image)  # [1, 1, 384, 384]
    probability = logits.sigmoid()
    mask = probability >= 0.5
```

Inputs are floating-point RGB tensors in `[0, 1]`. ImageNet normalization is performed inside the model; do not normalize inputs twice. The output is a single-channel foreground **logit** map. Reference height and width must be multiples of 32 and at least 64 pixels.

`model(image)` generates the enlarged and reduced branches from the reference tensor by bilinear interpolation. To preserve the original experiment's preprocessing, prepare all three branches using the original image loader and call:

```python
logits = model.forward_scales(image_l, image_m, image_s)
# image_l: [N, 3, 576, 576]; image_m: [N, 3, 384, 384];
# image_s: [N, 3, 192, 192]. All RGB, unnormalized and in [0, 1].
```

## Training objective

The fixed objective is:

```text
L = L_BCE + 0.25 L_Dice + 0.45 L_IoU + 0.15 L_Boundary + 0.5 L_UAL
```

BCE foreground pixels receive weight `1.05`. The boundary loss uses a `5 × 5` morphological gradient. UAL uses the cosine coefficient `(1 − cos(π × progress)) / 2`, where `progress` is the completed fraction of training, from 0 to 1.

```python
from msrsc_net import MSRSCLoss

model.train()
criterion = MSRSCLoss()
# images: [N, 3, 384, 384]; targets: [N, 1, 384, 384], binary 0/1.
# Use a batch size greater than one: global branches contain BatchNorm.
logits = model(images)
loss = criterion(logits, targets, progress=current_step / total_steps)
loss.backward()
```

## Existing checkpoints

`load_checkpoint(model, path)` accepts either release-format weights or the original full MSRSC-Net experiment state dict. It translates projection, fusion, decoder and SCEM names and loads all active parameters strictly. The original video-only temporal parameters are discarded because their operator is an identity for single-image inputs. Missing or unexpected active weights raise an error.

To save weights using the new names:

```python
torch.save(model.state_dict(), "msrsc_net.pth")
```

## Verification

The release contains **66,649,852 parameters**. With the same trained full-model checkpoint and identical branch tensors, CPU FP32 comparisons against the original experiment implementation produced identical logits and binary masks at reference sizes `384 × 384` and `64 × 96`. Nine loss-value and gradient comparisons (random, empty and full masks at three training fractions), plus a complete-model backward pass, also passed. Removing the inactive video operator removes 80 parameters without changing single-image predictions.

## Acknowledgements

The shared encoder is based on [PVTv2](https://github.com/whai362/PVT). The three-scale framework, MHSIU, RGPU and uncertainty-aware loss build on [ZoomNeXt](https://github.com/lartpang/ZoomNeXt). Please acknowledge the respective works when using those components.
