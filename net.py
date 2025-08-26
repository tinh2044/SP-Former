import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_ch, out_ch, bias=True, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias, groups=groups)


def conv3x3(in_ch, out_ch, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: B,C,H,W -> transpose to B,H,W,C
        b, c, h, w = x.shape
        x_t = x.permute(0, 2, 3, 1).contiguous()
        x_t = self.norm(x_t)
        return x_t.permute(0, 3, 1, 2).contiguous()


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, head_dim=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        if head_dim is None:
            assert dim % heads == 0
            head_dim = dim // heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim

        self.to_q = conv1x1(dim, inner_dim, bias=False)
        self.to_k = conv1x1(dim, inner_dim, bias=False)
        self.to_v = conv1x1(dim, inner_dim, bias=False)
        self.to_out = conv1x1(inner_dim, dim)

    def phi(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q = self.to_q(x).reshape(b, self.heads, self.head_dim, n)
        k = self.to_k(x).reshape(b, self.heads, self.head_dim, n)
        v = self.to_v(x).reshape(b, self.heads, self.head_dim, n)

        q = self.phi(q)
        k = self.phi(k)

        KV = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d n, b h d e -> b h e n", q, KV)

        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out


class GatedDSFFN(nn.Module):
    def __init__(self, dim, expansion=2.0, dw_kernel=3):
        super().__init__()
        hidden = max(1, int(dim * expansion))
        self.pw1 = conv1x1(dim, hidden * 2, bias=True)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=dw_kernel,
            padding=dw_kernel // 2,
            groups=hidden,
            bias=True,
        )
        self.pw2 = conv1x1(hidden, dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.pw1(x)
        y1, y2 = y.chunk(2, dim=1)
        y1 = self.act(y1)
        y1 = self.dw(y1)
        y = y1 * torch.sigmoid(y2)
        y = self.pw2(y)
        return y


class TransformerBlockLite(nn.Module):
    def __init__(self, dim, heads=4, head_dim=None, ffn_expansion=2.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = LinearAttention(dim, heads=heads, head_dim=head_dim)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDSFFN(dim, expansion=ffn_expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GroupCorr(nn.Module):
    def __init__(self, in_ch, groups=4, proj=True):
        super().__init__()
        self.groups = groups
        self.group_ch = in_ch // groups
        self.proj = proj
        if proj:
            self.to_q = conv1x1(in_ch, in_ch, bias=False)
            self.to_k = conv1x1(in_ch, in_ch, bias=False)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, feat, guidance):
        b, c, h, w = feat.shape
        g = self.groups
        fq = self.to_q(guidance) if self.proj else guidance
        fk = self.to_k(feat) if self.proj else feat
        fq = fq.reshape(b, g, self.group_ch, h * w)
        fk = fk.reshape(b, g, self.group_ch, h * w)
        desc = fq.mean(-1, keepdim=True)
        sim = torch.einsum("bgci,bgcn->bgin", desc, fk) / (
            math.sqrt(self.group_ch) * self.temperature
        )
        attn = F.softmax(sim, dim=-1)
        out_g = torch.einsum("bgin,bgcn->bgci", attn, fk)
        out = out_g.reshape(b, c, 1, 1)
        out = out.expand(-1, -1, h, w)
        return out


class SpectralBankModule(nn.Module):
    def __init__(
        self, feat_ch, guidance_ch=3, bands=4, bank_kernel=3, groups=4, share_fuse=False
    ):
        super().__init__()
        self.bands = bands
        self.feat_ch = feat_ch
        self.groups = groups
        self.banks = nn.ModuleList()
        for _ in range(bands):
            self.banks.append(
                nn.Sequential(
                    nn.Conv2d(
                        guidance_ch,
                        guidance_ch,
                        kernel_size=bank_kernel,
                        padding=bank_kernel // 2,
                        groups=guidance_ch,
                        bias=False,
                    ),
                    conv1x1(guidance_ch, feat_ch),
                )
            )
        self.mask_conv = nn.Sequential(
            conv3x3(feat_ch * bands, feat_ch, bias=True),
            nn.ReLU(inplace=True),
            conv1x1(feat_ch, feat_ch),
        )
        self.separable_fuse = nn.Sequential(
            nn.Conv2d(
                feat_ch, feat_ch, kernel_size=3, padding=1, groups=feat_ch, bias=False
            ),
            conv1x1(feat_ch, feat_ch, bias=True),
        )
        self.groupcorr = GroupCorr(feat_ch, groups=groups, proj=True)

    def forward(self, feat, guidance):
        guidance_resized = F.interpolate(
            guidance, size=feat.shape[2:], mode="bilinear", align_corners=False
        )

        band_outs = []
        for bank in self.banks:
            r = bank(guidance_resized)
            band_outs.append(r)
        cat = torch.cat(band_outs, dim=1)
        masks = self.mask_conv(cat)  # B, feat_ch, H, W

        energies = torch.stack(
            [m.mean(dim=1, keepdim=True) for m in band_outs], dim=1
        )  # B, bands, 1, H, W
        weights = F.softmax(
            energies.view(energies.shape[0], energies.shape[1], -1).mean(-1), dim=1
        )
        weights = weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
        banks_stack = torch.stack(band_outs, dim=1)  # B,bands,C,H,W
        recomposed = (weights * banks_stack).sum(dim=1)  # B,C,H,W

        fused = recomposed * torch.sigmoid(masks)

        corr = self.groupcorr(feat, recomposed)
        fused = fused + corr

        fused = self.separable_fuse(fused)
        return fused


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, out_ch=36):
        super().__init__()
        self.proj = conv3x3(in_ch, out_ch)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Upsample + conv to reduce params (instead of convtranspose)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class SPFormer(nn.Module):
    def __init__(
        self,
        base_ch=20,
        blocks=[2, 2, 2, 2],
        heads=[2, 4, 8, 8],
        ffn_expansion=1.5,
        bands=4,
        **kwargs,
    ):
        super().__init__()
        d = base_ch

        c1, c2, c3, c4 = d, d * 2, d * 4, d * 8
        self.patch = PatchEmbed(3, c1)

        self.enc1 = nn.Sequential(
            *[
                TransformerBlockLite(c1, heads=heads[0], ffn_expansion=ffn_expansion)
                for _ in range(blocks[0])
            ]
        )
        self.down1 = Downsample(c1, c2)
        self.enc2 = nn.Sequential(
            *[
                TransformerBlockLite(c2, heads=heads[1], ffn_expansion=ffn_expansion)
                for _ in range(blocks[1])
            ]
        )
        self.down2 = Downsample(c2, c3)
        self.enc3 = nn.Sequential(
            *[
                TransformerBlockLite(c3, heads=heads[2], ffn_expansion=ffn_expansion)
                for _ in range(blocks[2])
            ]
        )
        self.down3 = Downsample(c3, c4)

        self.latent_blocks = nn.Sequential(
            *[
                TransformerBlockLite(c4, heads=heads[3], ffn_expansion=ffn_expansion)
                for _ in range(blocks[3])
            ]
        )

        self.sbm_latent = SpectralBankModule(c4, guidance_ch=3, bands=bands, groups=4)

        self.up3 = Upsample(c4, c3)
        self.dec3 = nn.Sequential(
            *[
                TransformerBlockLite(c3, heads=heads[2], ffn_expansion=ffn_expansion)
                for _ in range(blocks[2])
            ]
        )
        self.sbm3 = SpectralBankModule(c3, guidance_ch=3, bands=bands, groups=4)

        self.up2 = Upsample(c3, c2)
        self.dec2 = nn.Sequential(
            *[
                TransformerBlockLite(c2, heads=heads[1], ffn_expansion=ffn_expansion)
                for _ in range(blocks[1])
            ]
        )
        self.sbm2 = SpectralBankModule(c2, guidance_ch=3, bands=bands, groups=4)

        self.up1 = Upsample(c2, c1)
        self.dec1 = nn.Sequential(
            *[
                TransformerBlockLite(c1, heads=heads[0], ffn_expansion=ffn_expansion)
                for _ in range(blocks[0])
            ]
        )
        self.sbm1 = SpectralBankModule(c1, guidance_ch=3, bands=bands, groups=4)

        self.final_conv = conv3x3(c1, 3)

    def forward(self, x):
        bgr = x
        p = self.patch(x)
        e1 = self.enc1(p)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        latent = self.latent_blocks(d3)
        sbm_lat = self.sbm_latent(latent, bgr)
        latent = latent + sbm_lat

        u3 = self.up3(latent) + e3
        d3r = self.dec3(u3)
        sbm3 = self.sbm3(d3r, bgr)
        d3r = d3r + sbm3

        u2 = self.up2(d3r) + e2
        d2r = self.dec2(u2)
        sbm2 = self.sbm2(d2r, bgr)
        d2r = d2r + sbm2

        u1 = self.up1(d2r) + e1
        d1r = self.dec1(u1)
        sbm1 = self.sbm1(d1r, bgr)
        d1r = d1r + sbm1

        out = self.final_conv(d1r)
        # residual add to input RGB
        out = torch.clamp(out + x, 0.0, 1.0)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick smoke test and parameter print
    model = SPFormer(
        base_ch=20, blocks=[2, 2, 2, 2], heads=[2, 4, 8, 8], ffn_expansion=1.5, bands=4
    )
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)
    print("Params:", count_parameters(model))
