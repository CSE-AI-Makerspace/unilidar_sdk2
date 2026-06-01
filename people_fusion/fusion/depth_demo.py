#!/usr/bin/env python3
"""Direction-B preview: monocular metric depth from one camera frame (runs in container).

Runs a Depth Anything V2 metric model on an image and saves the depth map (.npy) plus a
colored visualization. This is the camera-only 3D estimate *before* LiDAR supervision; in
the full pipeline the LiDAR would correct its scale / fine-tune it.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-depth", default="/out/depth.npy")
    ap.add_argument("--out-color", default="/out/depth_color.png")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    a = ap.parse_args()

    dev = 0 if torch.cuda.is_available() else -1
    print(f"[depth] model={a.model} device={dev}", flush=True)
    pipe = pipeline("depth-estimation", model=a.model, device=dev)

    img = Image.open(a.image).convert("RGB")
    out = pipe(img)
    pd = out["predicted_depth"]
    depth = (pd.squeeze().detach().cpu().numpy() if hasattr(pd, "detach") else np.asarray(pd).squeeze()).astype("float32")

    w, h = img.size
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h))

    np.save(a.out_depth, depth)
    print(f"[depth] meters: min={depth.min():.2f} max={depth.max():.2f} mean={depth.mean():.2f} shape={depth.shape}", flush=True)
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    cv2.imwrite(a.out_color, cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO))
    print(f"[depth] wrote {a.out_depth} and {a.out_color}", flush=True)


if __name__ == "__main__":
    main()
