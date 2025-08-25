import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pytorch_msssim import SSIM as LossSSIM


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x, gt, feature_layers=[0, 1, 2, 3], style_layers=[]):
        x = (x - self.mean) / self.std
        gt = (gt - self.mean) / self.std
        if self.resize:
            x = self.transform(x, mode="bilinear", size=(224, 224), align_corners=False)
            gt = self.transform(
                gt, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        for i, block in enumerate(self.blocks):
            x = block(x)
            gt = block(gt)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, gt)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_gt = gt.reshape(gt.shape[0], gt.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_gt @ act_gt.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()

        kernel_g = [
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x, xx):
        grad = 0
        y = x
        yy = xx
        gradient_x = F.conv2d(y, self.weight_g, groups=3)
        gradient_xx = F.conv2d(yy, self.weight_g, groups=3)
        l = nn.L1Loss()
        a = l(gradient_x, gradient_xx)
        grad = grad + a
        return grad


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class UnderwaterLosses(nn.Module):
    """
    Consolidates the requested losses:
      - L1 (reconstruction)
      - Perceptual (VGG features)
      - MS-SSIM (multi-scale structural similarity)
      - Edge/Gradient loss (horizontal + vertical diffs)
    Optional: TV regularizer on t or A.
    Usage: loss_fn = UnderwaterLosses(config); total, comps = loss_fn(hatJ, J=..., I=..., t=..., A=...)
    """

    def __init__(
        self,
        weights=None,
        **kwargs,
    ):
        super().__init__()
        default_weights = {
            "l1": 1.0,
            "perc": 0.01,
            "grad": 0.1,
            "ssim": 0.1,
        }
        if weights is None:
            weights = default_weights
        else:
            for k, v in default_weights.items():
                weights.setdefault(k, v)
        self.weights = weights

        self.vgg = VGGPerceptualLoss()
        self.grad = Gradient_Loss()
        self.charbonnier = CharbonnierLoss()
        self.ssim = LossSSIM()

    def forward(self, pred, gt):
        """
        pred: Bx3xHxW predicted enhanced image (0..1)
        gt:   Bx3xHxW ground-truth clean image (0..1), optional for unpaired training
        returns: total_loss (scalar), dict of components
        """
        if pred is None or gt is None:
            raise ValueError("pred and gt must be provided")

        l_perc = self.weights["perc"] * self.vgg(pred, gt)
        l_grad = self.weights["grad"] * self.grad(pred, gt)
        l_l1 = self.weights["l1"] * self.charbonnier(pred, gt)
        l_ssim = self.weights["ssim"] * self.ssim(pred, gt)

        total_loss = l_l1 + l_perc + l_grad + l_ssim

        return {
            "l1": l_l1,
            "perc": l_perc,
            "grad": l_grad,
            "ssim": l_ssim,
            "total": total_loss,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = UnderwaterLosses().to(device)

    B, C, H, W = 2, 3, 128, 128
    pred = torch.rand(B, C, H, W, device=device)
    gt = torch.rand(B, C, H, W, device=device)

    comps = loss_fn(pred, gt)
    for k, v in comps.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.item()}")
        else:
            print(f"{k}: {v}")
