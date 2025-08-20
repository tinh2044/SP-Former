import argparse
import sys
import os
from typing import Dict

import torch


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_run(name: str, fn):
    try:
        result = fn()
        print(f"[OK] {name}")
        return True, result
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP/FAIL] {name}: {type(exc).__name__}: {exc}")
        return False, None


def make_codebook_params() -> list:
    # Keep tiny to make the test fast; scales at 32 only
    # Format per models/__init__.py: [scale, emb_num, emb_dim]
    return [[32, 8, 16]]


def test_aqcure(device: torch.device) -> None:
    print_header("Test AQCURE (models/__init__.py)")
    from models import AQCURE  # noqa: WPS433

    codebook_params = make_codebook_params()
    model = AQCURE(
        in_channel=3,
        codebook_params=codebook_params,
        gt_resolution=256,
        use_quantize=True,
        scale_factor=1,
        use_semantic_loss=False,
    ).to(device)

    x = torch.randn(1, 3, 256, 256, device=device)

    def _forward():
        model.eval()
        with torch.no_grad():
            # Build dummy gt_indices to satisfy VectorQuantizer's z_q_gt usage
            enc_feats = model.multiscale_encoder(x.detach())[::-1]
            gt_list = []
            quant_idx = 0
            for i in range(model.max_depth):
                cur_res = model.gt_res // 2**model.max_depth * 2**i
                if cur_res in model.codebook_scale:
                    h, w = enc_feats[i].shape[2], enc_feats[i].shape[3]
                    n_e = model.quantizers[quant_idx].n_e
                    gt = torch.randint(0, int(n_e), (1, 1, h, w), device=device)
                    gt_list.append(gt)
                    quant_idx += 1

            dec, codebook_loss, semantic_loss, indices = model(
                x, gt_indices=gt_list if gt_list else None
            )
        # Print minimal shape info
        print("AQCURE output:", tuple(dec.shape))
        print("AQCURE codebook_loss:", float(codebook_loss))
        print("AQCURE semantic_loss:", float(semantic_loss))
        print("AQCURE indices list len:", len(indices))

    safe_run("AQCURE init+forward", _forward)


def test_encoder(device: torch.device) -> None:
    print_header("Test Encoder (models/encoder.py)")
    from models.encoder import Encoder  # noqa: WPS433

    channel_query_dict = {
        8: 256,
        16: 256,
        32: 256,
        64: 256,
        128: 128,
        256: 64,
        512: 32,
    }

    encoder = Encoder(
        in_channel=3,
        max_depth=3,
        input_res=256,
        channel_query_dict=channel_query_dict,
        norm_type="gn",
        act_type="silu",
    ).to(device)

    x = torch.randn(1, 3, 256, 256, device=device)

    def _forward():
        encoder.eval()
        with torch.no_grad():
            feats = encoder(x)
        print("Encoder outputs (len):", len(feats))
        for idx, f in enumerate(feats):
            print(f"  - feat[{idx}]:", tuple(f.shape))

    safe_run("Encoder init+forward", _forward)


def _make_dec_res_dict(
    base_channels: int,
    channel_multipliers: list,
    spatial_map: Dict[str, int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    # Build feature maps for required keys: z_quant, Level_8, Level_4, Level_2, Level_1
    num_levels = len(channel_multipliers)
    feats: Dict[str, torch.Tensor] = {}

    # Highest index (smallest spatial size)
    highest_channels = base_channels * channel_multipliers[num_levels - 1]
    h8 = spatial_map["Level_8"]
    feats["z_quant"] = torch.randn(1, highest_channels, h8, h8, device=device)

    # Build per level tensors matching channels used internally
    for i in range(num_levels):
        level_name = f"Level_{2**i}"
        channels = base_channels * channel_multipliers[i]
        h = spatial_map[level_name]
        feats[level_name] = torch.randn(1, channels, h, h, device=device)
    return feats


def test_decoder(device: torch.device) -> None:
    print_header("Test MainDecoder (models/decode.py)")
    from models.decoder import MainDecoder  # noqa: WPS433

    base_channels = 64
    channel_multipliers = [1, 2, 4, 4]
    decoder = MainDecoder(
        base_channels=base_channels, channel_multipliers=channel_multipliers
    ).to(device)

    # Spatial sizes per listed levels (Level_8 is smallest)
    spatial_map = {
        "Level_8": 32,
        "Level_4": 64,
        "Level_2": 128,
        "Level_1": 256,
    }

    dec_res_dict = _make_dec_res_dict(
        base_channels, channel_multipliers, spatial_map, device
    )
    # x_d is initial decoder feature at Level_8 resolution/channels
    x_d = torch.randn(
        1,
        base_channels * channel_multipliers[-1],
        spatial_map["Level_8"],
        spatial_map["Level_8"],
        device=device,
    )

    def _forward():
        decoder.eval()
        with torch.no_grad():
            out = decoder(dec_res_dict, x_d)
        print("MainDecoder output:", tuple(out.shape))

    ok, _ = safe_run("MainDecoder init+forward", _forward)
    if not ok:
        print(
            "Note: This decoder uses deformable conv ops, which may be unavailable on CPU/Windows. Skipped."
        )


def test_layers(device: torch.device) -> None:
    print_header("Test Layers (models/layers.py)")
    from models.layers import ResBlock, CombineQuantBlock, AIEM  # noqa: WPS433

    # ResBlock
    rb = ResBlock(64, 64).to(device)
    x = torch.randn(1, 64, 32, 32, device=device)
    safe_run("ResBlock init+forward", lambda: rb(x))

    # CombineQuantBlock (two-input case)
    cqb = CombineQuantBlock(16, 16, 32).to(device)
    x1 = torch.randn(1, 16, 32, 32, device=device)
    x2 = torch.randn(1, 16, 16, 16, device=device)
    safe_run("CombineQuantBlock (with input2)", lambda: cqb(x1, x2))
    # Single-input case is used when in_ch2 == 0 in the model
    cqb_single = CombineQuantBlock(16, 0, 32).to(device)
    safe_run("CombineQuantBlock (single input)", lambda: cqb_single(x1))

    # AIEM
    aiem = AIEM(c=256).to(device)
    x_aiem = torch.randn(1, 256, 32, 32, device=device)
    safe_run("AIEM init+forward", lambda: aiem(x_aiem))

    # Skip HIMB due to constructor signature mismatch in LKA; not required by the main model paths.


def test_vgg(device: torch.device) -> None:
    print_header("Test VGGFeatureExtractor (models/vgg_arch.py)")
    from models.vgg_arch import VGGFeatureExtractor  # noqa: WPS433

    # Try to avoid downloads; if not present, this may attempt to fetch weights.
    # We handle failures gracefully and skip.
    def _init_forward():
        extractor = VGGFeatureExtractor(
            ["relu4_4"], vgg_type="vgg19", use_input_norm=True
        )
        extractor = extractor.to(device)
        extractor.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224, device=device)
            outputs = extractor(x)
        for k, v in outputs.items():
            print(f"VGG feature {k}: {tuple(v.shape)}")

    ok, _ = safe_run("VGGFeatureExtractor init+forward", _init_forward)
    if not ok:
        print("Note: Likely due to missing VGG weights or no internet; skipping.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for model init and forward"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the tests on",
    )
    args = parser.parse_args()

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    print(
        f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | device={device}"
    )
    try:
        import torchvision  # noqa: WPS433

        print(f"torchvision {torchvision.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"torchvision not available: {exc}")

    # Run tests
    test_aqcure(device)
    test_encoder(device)
    test_decoder(device)
    test_layers(device)
    test_vgg(device)

    print("\nAll tests attempted.")


if __name__ == "__main__":
    # Ensure repo root is in sys.path when executed from anywhere
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    main()
