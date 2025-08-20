import argparse
import os
import glob
import math
from typing import List

import cv2
import torch
import torch.nn.functional as F

from model import UIE_model
import yaml


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(input_path: str) -> List[str]:
    if os.path.isfile(input_path) and is_image_file(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(paths)
    return []


def pad_to_multiple(x: torch.Tensor, multiple: int) -> (torch.Tensor, int, int):
    # x: 1xCxHxW, pad right/bottom to be divisible by multiple (e.g., 8)
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    # Pad order for F.pad is (left, right, top, bottom)
    xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return xp, pad_h, pad_w


def crop_to_original(x: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    return x[:, :, :orig_h, :orig_w]


def load_cfg_model(cfg_path: str, device: torch.device) -> UIE_model:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    model = UIE_model(**model_cfg).to(device)
    model.eval()
    return model


def clean_state_dict_keys(state_dict):
    # Remove possible prefixes like 'module.' or 'model.'
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module.") :]] = v
        elif k.startswith("model."):
            new_sd[k[len("model.") :]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_weights(model: UIE_model, weight_path: str):
    ckpt = torch.load(weight_path, map_location="cpu")
    # Try common formats
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            # maybe the dict is already state_dict
            sd = ckpt
    else:
        # Unexpected format
        sd = ckpt
    sd = clean_state_dict_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print("[Warning] Missing keys:", missing)
    if len(unexpected) > 0:
        print("[Warning] Unexpected keys:", unexpected)


def enhance_image(
    model: UIE_model, img_bgr: cv2.Mat, device: torch.device, pad_multiple: int = 8
) -> cv2.Mat:
    # Convert BGR uint8 -> RGB float tensor in [0,1]
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype("float32") / 255.0
    t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)  # 1x3xHxW

    orig_h, orig_w = t.shape[2], t.shape[3]
    t, pad_h, pad_w = pad_to_multiple(t, pad_multiple)

    with torch.no_grad():
        out_dict = model(t)
        out = out_dict["output"].clamp(0.0, 1.0)

    # Crop back to original size if padded
    if pad_h != 0 or pad_w != 0:
        out = crop_to_original(out, orig_h, orig_w)

    out_np = (
        (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        .round()
        .clip(0, 255)
        .astype("uint8")
    )
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr


def main():
    parser = argparse.ArgumentParser("UIE_model Inference")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input image or folder"
    )
    parser.add_argument(
        "-w", "--weight", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./results_uie", help="Output folder"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/uieb.yaml",
        help="Path to config (model params)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--suffix", type=str, default="_uie", help="Suffix for saved images"
    )
    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=8,
        help="Pad H,W to be divisible by this value",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Build model and load weights
    model = load_cfg_model(args.cfg_path, device)
    load_weights(model, args.weight)
    model.eval()

    # Collect images
    paths = list_images(args.input)
    if len(paths) == 0:
        print(f"No images found in {args.input}")
        return

    for path in paths:
        img_name = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Warning] Failed to read image: {path}")
            continue

        res = enhance_image(model, img, device, pad_multiple=args.pad_multiple)

        name, ext = os.path.splitext(img_name)
        save_name = f"{name}{args.suffix}{ext}"
        save_path = os.path.join(args.output, save_name)
        cv2.imwrite(save_path, res)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
