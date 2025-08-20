import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips
import warnings

# Initialize LPIPS once; suppress pickle FutureWarning from inside the library load
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
    )
    _lpips_model = lpips.LPIPS(net="vgg")


def calculate_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2):
    # SSIM expects (B, C, H, W)
    return ssim(img1, img2, data_range=1.0, size_average=True).item()


def calculate_lpips(img1, img2, device="cuda"):
    model = _lpips_model.to(device)
    with torch.no_grad():
        return model(img1, img2).mean().item()


def compute_metrics(img1, img2, device="cuda"):
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    lpips = calculate_lpips(img1, img2, device)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }
