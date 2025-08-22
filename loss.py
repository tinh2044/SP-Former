import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pytorch_msssim import ms_ssim as pytorch_ms_ssim


class VGGFeatureExtractor(nn.Module):
    """
    Simple VGG19 feature extractor returning a list of intermediate activations.
    Layers spec: accept list of layer names among ['relu1_1','relu1_2','relu2_1',...,'relu5_4'].
    """

    def __init__(self, layers=("relu3_3", "relu4_3"), requires_grad=False, device=None):
        super().__init__()
        vgg_pretrained = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1
        ).features.eval()
        self.layer_name_mapping = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 17,
            "relu4_1": 20,
            "relu4_2": 22,
            "relu4_3": 24,
            "relu4_4": 26,
            "relu5_1": 29,
            "relu5_2": 31,
            "relu5_3": 33,
            "relu5_4": 35,
        }
        max_idx = max(self.layer_name_mapping[l] for l in layers)
        self.vgg_slices = nn.Sequential(
            *[vgg_pretrained[i] for i in range(max_idx + 1)]
        )
        self.requested_layers = layers
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False
        if device is not None:
            self.to(device=device)

    def forward(self, x):
        # Ensure input is in valid range and add small epsilon to prevent division by zero
        x = torch.clamp(x, 0.0, 1.0)

        # ImageNet normalization - handle both [0,1] and [-1,1] ranges
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )

        # Normalize to [0,1] first if input is in [-1,1]
        if x.min() < 0:
            x = (x + 1) / 2

        # Add small epsilon to prevent extreme values
        x_in = (x - mean) / (std + 1e-8)

        out_feats = []
        cur = x_in
        for idx, layer in enumerate(self.vgg_slices):
            cur = layer(cur)
            for name, i in self.layer_name_mapping.items():
                if i == idx and name in self.requested_layers:
                    out_feats.append(cur)
        return out_feats


class UnderwaterLosses(nn.Module):
    """
    Consolidates the requested losses:
      - L1 (reconstruction)
      - Perceptual (VGG features)
      - MS-SSIM (multi-scale structural similarity)
      - Physics-consistency (I ≈ J * t + A * (1-t))
      - Edge/Gradient loss (horizontal + vertical diffs)
    Optional: TV regularizer on t or A.
    Usage: loss_fn = UnderwaterLosses(config); total, comps = loss_fn(hatJ, J=..., I=..., t=..., A=...)
    """

    def __init__(
        self,
        weights=None,
        perc_layers=("relu3_3", "relu4_3"),
        ms_window_size=11,
        ms_window_sigma=1.5,
        use_tv_on_t=True,
        tv_weight=1e-3,
        device=None,
    ):
        super().__init__()
        default_weights = {
            "l1": 1.0,
            "perc": 0.01,  # Reduced perceptual weight to prevent exploding
            "ms_ssim": 0.5,
            "phys": 2.0,  # Increased physics weight for better underwater modeling
            "grad": 0.1,
            "tv": 1e-3,  # TV regularization weight
        }
        if weights is None:
            weights = default_weights
        else:
            for k, v in default_weights.items():
                weights.setdefault(k, v)
        self.weights = weights
        self.ms_window_size = ms_window_size
        self.ms_window_sigma = ms_window_sigma
        self.perc_layers = perc_layers
        self.use_tv_on_t = use_tv_on_t
        self.tv_weight = tv_weight
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.vgg = VGGFeatureExtractor(
            layers=perc_layers, requires_grad=False, device=self.device
        )

    def forward(self, hatJ, J=None, I=None, t=None, A=None):
        """
        hatJ: Bx3xHxW predicted enhanced image (0..1)
        J:    Bx3xHxW ground-truth clean image (0..1), optional for unpaired training
        I:    Bx3xHxW original underwater image (0..1), optional (used for physics loss)
        t:    predicted transmission (B x 3 x H x W) or None
        A:    predicted veiling (B x 3 x H x W) or None
        returns: total_loss (scalar), dict of components
        """
        device = self.device
        hatJ = hatJ.to(device)

        # Add safeguards to prevent NaN
        hatJ = torch.clamp(hatJ, 0.0, 1.0)
        if J is not None:
            J = J.to(device)
            J = torch.clamp(J, 0.0, 1.0)
        if I is not None:
            I = I.to(device)
            I = torch.clamp(I, 0.0, 1.0)
        if t is not None:
            t = t.to(device)
            t = torch.clamp(t, 0.01, 0.99)  # Avoid extreme values
        if A is not None:
            A = A.to(device)
            A = torch.clamp(A, 0.0, 1.0)

        # Denormalize from [-1, 1] to [0, 1] if needed (for models that expect [0, 1])
        if hatJ.min() < 0:
            hatJ = (hatJ + 1) / 2
        if J is not None and J.min() < 0:
            J = (J + 1) / 2
        if I is not None and I.min() < 0:
            I = (I + 1) / 2

        comps = {}

        # L1 Loss
        if J is not None:
            l_l1 = F.l1_loss(hatJ, J, reduction="mean")
        else:
            l_l1 = torch.tensor(0.0, device=device)
        comps["l1"] = l_l1

        # Perceptual Loss
        if (self.vgg is not None) and (J is not None):
            try:
                # Ensure inputs are in valid range for VGG
                hatJ_perc = torch.clamp(hatJ, 0.0, 1.0)
                J_perc = torch.clamp(J, 0.0, 1.0)

                feats_hat = self.vgg(hatJ_perc)
                feats_gt = self.vgg(J_perc)

                l_perc = 0.0
                valid_features = 0
                for fh, fg in zip(feats_hat, feats_gt):
                    # Check for NaN or inf in features
                    if torch.isfinite(fh).all() and torch.isfinite(fg).all():
                        mse_loss = F.mse_loss(fh, fg, reduction="mean")
                        if torch.isfinite(mse_loss):
                            l_perc = l_perc + mse_loss
                            valid_features += 1

                # Only use perceptual loss if we have valid features
                if valid_features > 0:
                    comps["perc"] = l_perc
                else:
                    print("No valid features from VGG, using fallback")
                    comps["perc"] = torch.tensor(0.0, device=device)

            except Exception as e:
                print(f"Perceptual loss error: {e}")
                comps["perc"] = torch.tensor(0.0, device=device)
        else:
            comps["perc"] = torch.tensor(0.0, device=device)

        # MS-SSIM Loss
        if J is not None:
            try:
                # Ensure inputs are valid for MS-SSIM
                hatJ_ms = torch.clamp(hatJ, 0.0, 1.0)
                J_ms = torch.clamp(J, 0.0, 1.0)

                # Use pytorch_msssim directly
                ms = pytorch_ms_ssim(
                    hatJ_ms,
                    J_ms,
                    data_range=1.0,
                    size_average=True,  # Return per-sample scores
                    win_size=self.ms_window_size,
                    win_sigma=self.ms_window_sigma,
                    weights=None,  # Use default weights
                )

                # Check for NaN in MS-SSIM output
                if torch.isnan(ms).any():
                    print("MS-SSIM output contains NaN, using fallback")
                    l_ms = torch.tensor(0.0, device=device)
                else:
                    l_ms = (1.0 - ms).mean()

                comps["ms_ssim"] = l_ms
            except Exception as e:
                print(f"MS-SSIM loss error: {e}")
                comps["ms_ssim"] = torch.tensor(0.0, device=device)
        else:
            comps["ms_ssim"] = torch.tensor(0.0, device=device)

        # Physics Loss
        if (I is not None) and (t is not None) and (A is not None):
            try:
                # Ensure physics parameters match image dimensions
                if t.shape[-2:] != I.shape[-2:]:
                    t = F.interpolate(
                        t, size=I.shape[-2:], mode="bilinear", align_corners=False
                    )
                if A.shape[-2:] != I.shape[-2:]:
                    A = F.interpolate(
                        A, size=I.shape[-2:], mode="bilinear", align_corners=False
                    )

                I_model = hatJ * t + A * (1.0 - t)
                l_phys = F.l1_loss(I_model, I, reduction="mean")
                comps["phys_rec"] = l_phys
                if self.use_tv_on_t:
                    tv_t = UnderwaterLosses.total_variation(t)
                    tv_A = UnderwaterLosses.total_variation(A)
                    comps["tv_t"] = tv_t
                    comps["tv_A"] = tv_A
                else:
                    comps["tv_t"] = torch.tensor(0.0, device=device)
                    comps["tv_A"] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Physics loss error: {e}")
                comps["phys_rec"] = torch.tensor(0.0, device=device)
                comps["tv_t"] = torch.tensor(0.0, device=device)
                comps["tv_A"] = torch.tensor(0.0, device=device)
        else:
            comps["phys_rec"] = torch.tensor(0.0, device=device)
            comps["tv_t"] = torch.tensor(0.0, device=device)
            comps["tv_A"] = torch.tensor(0.0, device=device)

        # Gradient Loss
        if J is not None:
            try:

                def grad_x(img):
                    return img[:, :, :, :-1] - img[:, :, :, 1:]

                def grad_y(img):
                    return img[:, :, :-1, :] - img[:, :, 1:, :]

                gx_hat = grad_x(hatJ)
                gy_hat = grad_y(hatJ)
                gx_gt = grad_x(J)
                gy_gt = grad_y(J)
                l_grad = (
                    F.l1_loss(gx_hat, gx_gt, reduction="mean")
                    + F.l1_loss(gy_hat, gy_gt, reduction="mean")
                ) * 0.5
                comps["grad"] = l_grad
            except Exception as e:
                print(f"Gradient loss error: {e}")
                comps["grad"] = torch.tensor(0.0, device=device)
        else:
            comps["grad"] = torch.tensor(0.0, device=device)

        # Check for NaN and replace with zeros
        for k, v in comps.items():
            if isinstance(v, torch.Tensor) and torch.isnan(v):
                print(f"NaN detected in {k}, replacing with 0")
                comps[k] = torch.tensor(0.0, device=device)

        # Compute total loss
        total = (
            self.weights["l1"] * comps["l1"]
            + self.weights["perc"] * comps["perc"]
            + self.weights["ms_ssim"] * comps["ms_ssim"]
            + self.weights["phys"] * comps["phys_rec"]
            + self.weights["grad"] * comps["grad"]
        )
        if self.use_tv_on_t:
            total = total + self.weights.get("tv", 0.0) * (
                comps["tv_t"] + comps["tv_A"]
            )

        # Final NaN check
        if torch.isnan(total):
            print("Total loss is NaN, using L1 only")
            total = self.weights["l1"] * comps["l1"]

        comps["total"] = total
        return total, comps

    @staticmethod
    def total_variation(x):
        # x: BxCxHxW
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = UnderwaterLosses(device=device, use_tv_on_t=True).to(device)

    B, C, H, W = 2, 3, 128, 128
    hatJ = torch.rand(B, C, H, W, device=device)
    J = torch.rand(B, C, H, W, device=device)
    I = torch.rand(B, C, H, W, device=device)
    t = (
        torch.sigmoid(torch.rand(B, C, H, W, device=device)) * 0.9 + 0.05
    )  # in (0.05,0.95)
    A = torch.sigmoid(torch.rand(B, C, H, W, device=device))

    comps = loss_fn(hatJ, J=J, I=I, t=t, A=A)
    for k, v in comps.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.item()}")
        else:
            print(f"{k}: {v}")
