import argparse
import os
import glob
from typing import List
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from net import SPFormer
import yaml


def is_image_file(path: str) -> bool:
    """Check if file is a valid image"""
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(input_path: str) -> List[str]:
    """Get list of all image files in path"""
    if os.path.isfile(input_path) and is_image_file(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(paths)
    return []


def load_model_from_config(cfg_path: str, device: torch.device) -> SPFormer:
    """Load model from config file like main.py"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Create model with config parameters
    model_cfg = cfg.get("model", {})
    model = SPFormer(**model_cfg)
    model = model.to(device)
    model.eval()

    print(
        f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model, cfg


def load_checkpoint(model: SPFormer, weight_path: str):
    """Load model weights from checkpoint"""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean state dict keys (remove 'module.' prefix if any)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    # Load with strict=False to handle potential key mismatches
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    print(f"Weights loaded from: {weight_path}")


def preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Preprocess image for model input"""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Define transforms (same as dataset.py)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Apply transforms
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """Convert model output tensor back to PIL Image"""
    # Denormalize and convert to numpy
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and then PIL Image
    array = tensor.permute(1, 2, 0).cpu().numpy()
    array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array)

    return image


def enhance_image(
    model: SPFormer, image_path: str, device: torch.device, image_size: int = 256
) -> Image.Image:
    """Enhance single image using the model"""
    # Preprocess
    input_tensor = preprocess_image(image_path, image_size).to(device)

    # Forward pass
    with torch.no_grad():
        output_dict = model(input_tensor)
        output_tensor = output_dict["output"]

    # Postprocess
    enhanced_image = postprocess_image(output_tensor)

    return enhanced_image


def process_batch(
    model: SPFormer,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    image_size: int = 256,
):
    """Process all images in input directory"""
    # Get all image files
    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process")

    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        try:
            print(f"Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")

            # Enhance image
            enhanced_image = enhance_image(model, image_path, device, image_size)

            # Save enhanced image
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
            enhanced_image.save(output_path)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print(f"Processing completed! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser("SPFormer Inference")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input image or directory containing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory to save enhanced images",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/uieb.yaml",
        help="Path to config file (default: configs/uieb.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size for processing (default: 256)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    try:
        # Load model from config (same as main.py)
        print("Loading model...")
        model, cfg = load_model_from_config(args.cfg_path, device)

        # Load weights
        print("Loading checkpoint...")
        load_checkpoint(model, args.weight)

        # Get image size from config or use default
        image_size = cfg.get("data", {}).get("image_size", args.image_size)

        # Process images
        process_batch(model, args.input, args.output, device, image_size)

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
