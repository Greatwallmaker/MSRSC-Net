# MSRSC-Net

PyTorch implementation of **Multiscale Reliability and Structural Continuity for Semantic Segmentation of Great Wall Remains from UAV Imagery**. MSRSC-Net segments Great Wall remains using RGB UAV orthophoto images.

## Installation

Python 3.10 or later is required. Install PyTorch for your CPU/CUDA environment, then run:

```bash
git clone https://github.com/Greatwallmaker/MSRSC-Net.git
cd MSRSC-Net
pip install -r requirements.txt
```

## Inference

Run inference on an image or a folder of image patches:

```bash
python predict.py --weights path/to/model.pth --input path/to/images --output outputs
```

## Acknowledgements

Our implementation builds on [ZoomNeXt](https://github.com/lartpang/ZoomNeXt) for the three-scale framework, MHSIU and RGPU, and [PVTv2](https://github.com/whai362/PVT) for the shared encoder. We thank the authors for sharing their work. The PVT license is retained [here](licenses/PVT-Apache-2.0.txt).
