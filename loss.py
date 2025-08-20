import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _gaussian(window_size: int, sigma: float):
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def create_window(window_size: int, channel: int, sigma: float = 1.5, device=None):
    _1D_window = _gaussian(window_size, sigma).to(device=device)
    _2D_window = _1D_window[:, None] @ _1D_window[None, :]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_map(img1, img2, window, window_size, channel, C1, C2):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    s_map = numerator / (denominator + 1e-12)
    return s_map.clamp(0.0, 1.0)


def ms_ssim(
    img1, img2, data_range=1.0, window_size=11, window_sigma=1.5, weights=None, levels=5
):
    device = img1.device
    B, C, H, W = img1.shape
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
    else:
        weights = torch.tensor(weights, device=device)
    window = create_window(window_size, C, sigma=window_sigma, device=device)

    L = data_range
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mssim = []
    mcs = []
    img1_scale = img1
    img2_scale = img2
    for l in range(levels):
        s_map = ssim_map(img1_scale, img2_scale, window, window_size, C, C1, C2)
        mean_s = s_map.mean(dim=[1, 2, 3])  # B
        mu1 = F.conv2d(img1_scale, window, padding=window_size // 2, groups=C)
        mu2 = F.conv2d(img2_scale, window, padding=window_size // 2, groups=C)
        sigma1_sq = F.conv2d(
            img1_scale * img1_scale, window, padding=window_size // 2, groups=C
        ) - mu1.pow(2)
        sigma2_sq = F.conv2d(
            img2_scale * img2_scale, window, padding=window_size // 2, groups=C
        ) - mu2.pow(2)
        sigma12 = (
            F.conv2d(
                img1_scale * img2_scale, window, padding=window_size // 2, groups=C
            )
            - mu1 * mu2
        )
        cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2 + 1e-12)
        mean_cs = cs_map.mean(dim=[1, 2, 3])

        mssim.append(mean_s)
        mcs.append(mean_cs)

        img1_scale = F.avg_pool2d(
            img1_scale, kernel_size=2, stride=2, count_include_pad=False
        )
        img2_scale = F.avg_pool2d(
            img2_scale, kernel_size=2, stride=2, count_include_pad=False
        )

    mssim = torch.stack(mssim, dim=0)  # levels x B
    mcs = torch.stack(mcs, dim=0)

    weights = weights[: mssim.size(0)].unsqueeze(1)  # Lx1
    ms = (mcs[:-1] ** weights[:-1]).prod(dim=0) * (mssim[-1] ** weights[-1])
    return ms  # return per-sample score


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
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_in = (x - mean) / std
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
        ms_ssim_levels=5,
        ms_window_size=11,
        ms_window_sigma=1.5,
        use_tv_on_t=False,
        tv_weight=1e-4,
        device=None,
    ):
        super().__init__()
        default_weights = {
            "l1": 1.0,
            "perc": 0.02,
            "ms_ssim": 0.5,
            "phys": 0.8,
            "grad": 0.1,
            "tv": tv_weight,
        }
        if weights is None:
            weights = default_weights
        else:
            for k, v in default_weights.items():
                weights.setdefault(k, v)
        self.weights = weights
        self.ms_levels = ms_ssim_levels
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
                feats_hat = self.vgg(hatJ)
                feats_gt = self.vgg(J)
                l_perc = 0.0
                for fh, fg in zip(feats_hat, feats_gt):
                    l_perc = l_perc + F.mse_loss(fh, fg, reduction="mean")
                comps["perc"] = l_perc
            except Exception as e:
                print(f"Perceptual loss error: {e}")
                comps["perc"] = torch.tensor(0.0, device=device)
        else:
            comps["perc"] = torch.tensor(0.0, device=device)

        # MS-SSIM Loss
        if J is not None:
            try:
                ms = ms_ssim(
                    hatJ,
                    J,
                    data_range=1.0,
                    window_size=self.ms_window_size,
                    window_sigma=self.ms_window_sigma,
                    levels=self.ms_levels,
                )
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
                I_model = hatJ * t + A * (1.0 - t)
                l_phys = F.l1_loss(I_model, I, reduction="mean")
                comps["phys_rec"] = l_phys
                if self.use_tv_on_t:
                    tv_t = self.total_variation(t)
                    tv_A = self.total_variation(A)
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
