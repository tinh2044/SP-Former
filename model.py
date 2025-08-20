import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from loss import UnderwaterLosses
except Exception:

    class UnderwaterLosses(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, pred, gt):
            if gt is None:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return F.l1_loss(pred, gt)


def conv3x3(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x1(in_ch, out_ch, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LinearizedAttention(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = conv1x1(dim, dim * 3, bias=bias)
        self.proj = conv1x1(dim, dim, bias=bias)

    def phi(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        n = h * w

        def reshape(t):
            return t.view(b, self.num_heads, self.head_dim, n)

        q_h, k_h, v_h = reshape(q), reshape(k), reshape(v)
        q_phi, k_phi = self.phi(q_h), self.phi(k_h)
        KV = torch.einsum("b h c n, b h d n -> b h c d", k_phi, v_h)
        out_h = torch.einsum("b h c n, b h c d -> b h d n", q_phi, KV)
        out = out_h.contiguous().view(b, c, h, w)
        return self.proj(out)


class GatedDSFFN(nn.Module):
    def __init__(self, dim, expansion=2.0, bias=False):
        super().__init__()
        hidden = int(dim * expansion)
        self.pw1 = conv1x1(dim, hidden * 2, bias=bias)
        self.dw = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.pw2 = conv1x1(hidden, dim, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        u = self.pw1(x)
        u = self.dw(u)
        a, b = u.chunk(2, dim=1)
        out = a * self.act(b)
        return self.pw2(out)


class TransformerBlockLite(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion=2.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = LinearizedAttention(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDSFFN(dim, expansion=ffn_expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            conv3x3(n_feat, n_feat // 2, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            conv3x3(n_feat, n_feat * 2, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=36):
        super().__init__()
        self.proj = conv3x3(in_ch, embed_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class GroupCorr(nn.Module):
    def __init__(self, dim, groups=4, bias=False):
        super().__init__()
        assert dim % groups == 0
        self.groups = groups
        self.dim = dim
        self.gdim = dim // groups
        self.q = conv1x1(dim, dim, bias=bias)
        self.k = conv1x1(dim, dim, bias=bias)
        self.v = conv1x1(dim, dim, bias=bias)
        self.out = conv1x1(dim, dim, bias=bias)
        self.norm = nn.Softmax(dim=-1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        n = h * w
        q = self.q(x).view(b, self.groups, self.gdim, n)
        k = self.k(y).view(b, self.groups, self.gdim, n)
        v = self.v(y).view(b, self.groups, self.gdim, n)
        k_desc = k.mean(-1, keepdim=True)
        # FIXED: use correct einsum with k_desc shape (B, g, c', 1)
        # Use 'd' for the single dimension instead of '1'
        att = torch.einsum("b g c n, b g c d -> b g d n", q, k_desc)
        att = self.norm(att.view(b, -1)).view(b, self.groups, 1, n)
        out = att * v
        out = out.view(b, c, h, w)
        return self.out(out)


class SpectralBankModule(nn.Module):
    def __init__(self, in_ch, feat_ch, bands=4, bias=False):
        super().__init__()
        self.proj_in = conv3x3(in_ch, feat_ch, bias=bias)
        self.bank = nn.Conv2d(
            feat_ch,
            feat_ch * bands,
            kernel_size=3,
            padding=1,
            groups=feat_ch,
            bias=bias,
        )
        self.mask_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(feat_ch, max(4, bands), bias=bias),
            nn.Sigmoid(),
        )
        self.fuse = conv1x1(feat_ch, feat_ch, bias=bias)
        self.beta_head = nn.Sequential(conv1x1(feat_ch, 3, bias=bias), nn.ReLU())
        self.A_head = nn.Sequential(conv1x1(feat_ch, 3, bias=bias), nn.Sigmoid())
        self.gcorr = GroupCorr(feat_ch, groups=4, bias=bias)
        self.jproj = conv1x1(3, feat_ch, bias=bias)
        self.bands = bands

    def forward(self, guidance, feat):
        b, c, hf, wf = feat.shape
        guidance_resize = F.interpolate(
            guidance, size=(hf, wf), mode="bilinear", align_corners=False
        )
        tilde = self.proj_in(guidance_resize)
        bank_resp = self.bank(tilde).view(b, tilde.size(1), self.bands, hf, wf)
        bank_mag = torch.abs(bank_resp)
        mask = self.mask_pred(tilde)
        if mask.shape[1] > self.bands:
            mask = mask[:, : self.bands, :, :]
        elif mask.shape[1] < self.bands:
            pad = torch.ones(
                b,
                self.bands - mask.shape[1],
                1,
                1,
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([mask, pad], dim=1)
        mask = mask.view(b, 1, self.bands, 1, 1)
        recomposed = (bank_mag * mask).sum(dim=2)
        recomposed = self.fuse(recomposed)
        beta = self.beta_head(tilde)
        A = self.A_head(tilde)
        eps = 1e-6
        if guidance_resize.shape[1] >= 3:
            I_rgb = guidance_resize[:, :3, :, :]
            t = torch.exp(-beta)
            J_approx = (I_rgb - A * (1 - t)) / (t + eps)
            Jdesc = F.adaptive_avg_pool2d(J_approx, 1)
            Jdesc = self.jproj(Jdesc)
            Jdesc = Jdesc.expand(-1, feat.size(1), hf, wf)
            fused = self.gcorr(recomposed, (feat + Jdesc))
        else:
            fused = self.gcorr(recomposed, feat)
        return fused


class UIE_model(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        d=36,
        num_blocks=[2, 2, 2, 3],
        num_refinement=2,
        heads=[1, 2, 2, 4],
        ffn_exp=2.0,
        loss_cfg=None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch=inp_channels, embed_dim=d)
        self.encoder1 = nn.Sequential(
            *[
                TransformerBlockLite(dim=d, num_heads=heads[0], ffn_expansion=ffn_exp)
                for _ in range(num_blocks[0])
            ]
        )
        self.down1 = Downsample(d)
        self.encoder2 = nn.Sequential(
            *[
                TransformerBlockLite(
                    dim=2 * d, num_heads=heads[1], ffn_expansion=ffn_exp
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.down2 = Downsample(2 * d)
        self.encoder3 = nn.Sequential(
            *[
                TransformerBlockLite(
                    dim=4 * d, num_heads=heads[2], ffn_expansion=ffn_exp
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.down3 = Downsample(4 * d)
        self.latent = nn.Sequential(
            *[
                TransformerBlockLite(
                    dim=8 * d, num_heads=heads[3], ffn_expansion=ffn_exp
                )
                for _ in range(num_blocks[3])
            ]
        )
        self.sbm4 = SpectralBankModule(in_ch=inp_channels, feat_ch=8 * d, bands=4)
        self.up4_3 = Upsample(8 * d)
        self.reduce3 = conv1x1(8 * d, 4 * d)
        self.decoder3 = nn.Sequential(
            *[
                TransformerBlockLite(
                    dim=4 * d, num_heads=heads[2], ffn_expansion=ffn_exp
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.sbm3 = SpectralBankModule(in_ch=inp_channels, feat_ch=4 * d, bands=4)
        self.up3_2 = Upsample(4 * d)
        self.reduce2 = conv1x1(4 * d, 2 * d)
        self.decoder2 = nn.Sequential(
            *[
                TransformerBlockLite(
                    dim=2 * d, num_heads=heads[1], ffn_expansion=ffn_exp
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.sbm2 = SpectralBankModule(in_ch=inp_channels, feat_ch=2 * d, bands=3)
        self.up2_1 = Upsample(2 * d)
        self.reduce1 = conv1x1(2 * d, d)
        self.decoder1 = nn.Sequential(
            *[
                TransformerBlockLite(dim=d, num_heads=heads[0], ffn_expansion=ffn_exp)
                for _ in range(num_blocks[0])
            ]
        )
        self.refine = nn.Sequential(
            *[
                TransformerBlockLite(dim=d, num_heads=heads[0], ffn_expansion=ffn_exp)
                for _ in range(num_refinement)
            ]
        )
        self.out_conv = conv3x3(d, out_channels, bias=False)
        self.loss_fn = UnderwaterLosses(**(loss_cfg if loss_cfg else {}))

    def forward(self, inp, gt=None):
        e1 = self.patch_embed(inp)
        en1 = self.encoder1(e1)
        d12 = self.down1(en1)
        en2 = self.encoder2(d12)
        d23 = self.down2(en2)
        en3 = self.encoder3(d23)
        d34 = self.down3(en3)
        lat = self.latent(d34)
        lat = lat + self.sbm4(inp, lat)
        u3 = self.up4_3(lat)
        cat3 = self.reduce3(torch.cat([u3, en3], dim=1))
        dec3 = self.decoder3(cat3)
        dec3 = dec3 + self.sbm3(inp, dec3)
        u2 = self.up3_2(dec3)
        cat2 = self.reduce2(torch.cat([u2, en2], dim=1))
        dec2 = self.decoder2(cat2)
        dec2 = dec2 + self.sbm2(inp, dec2)
        u1 = self.up2_1(dec2)
        cat1 = self.reduce1(torch.cat([u1, en1], dim=1))
        dec1 = self.decoder1(cat1)
        dec1 = self.refine(dec1)
        out = self.out_conv(dec1) + inp
        if gt is not None:
            loss = self.loss_fn(out, gt)
            return {"output": out, "loss": loss}
        return {"output": out}


if __name__ == "__main__":
    model = UIE_model(
        d=36, num_blocks=[2, 2, 2, 3], num_refinement=2, heads=[1, 2, 2, 4], ffn_exp=2.0
    )
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("UIE_model total trainable parameters:", p)
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    out = model(x, y)
    print("Forward OK. Output shape:", out["output"].shape)
