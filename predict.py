"""Command-line inference for RGB DOM image patches using trained MSRSC-Net weights."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from msrsc_net import MSRSCNet, load_checkpoint


def prepare_scales(rgb, device):
    """Match the original loader: resize each branch directly from the RGB image."""
    return [
        torch.from_numpy(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR))
        .float()
        .div(255)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        for size in (576, 384, 192)
    ]


@torch.inference_mode()
def predict(weights, source, output, device):
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if source.is_file() and source.suffix.lower() in extensions:
        images = [source]
    elif source.is_dir():
        images = sorted(
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
    else:
        raise ValueError(
            f"Expected an image file or a folder of image patches: {source}"
        )
    if not images:
        raise ValueError(f"No supported images found in {source}")
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    model = MSRSCNet()
    load_checkpoint(model, weights)
    model.to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images, 1):
        # imdecode/tofile also support Unicode paths on Windows.
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        logits = model.forward_scales(*prepare_scales(rgb, device))
        probability = logits.sigmoid()[0, 0].cpu().numpy()
        probability = cv2.resize(
            probability, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        mask = (probability >= 0.5).astype(np.uint8) * 255
        destination = output / f"{path.name}.mask.png"
        success, encoded = cv2.imencode(".png", mask)
        if not success:
            raise RuntimeError(f"Cannot encode mask for {path}")
        encoded.tofile(destination)
        print(f"[{index}/{len(images)}] {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, required=True, help="Trained MSRSC-Net checkpoint"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Image file or folder of RGB image patches",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="Output mask directory"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu, cuda, or cuda:0",
    )
    args = parser.parse_args()
    predict(args.weights, args.input, args.output, args.device)


if __name__ == "__main__":
    main()
