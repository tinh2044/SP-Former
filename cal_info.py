import os
from pathlib import Path
import torch
import argparse
from thop import profile, clever_format
import yaml

from net import SPFormer
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "SPFormer FLOPs and Parameters Calculator", add_help=False
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/lsui.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Image size for FLOPs calculation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for FLOPs calculation"
    )

    return parser


def count_module_flops(model, input_tensor, module_name):
    """Calculate FLOPs for a specific module"""
    try:
        # Handle different input types (single tensor or tuple)
        if isinstance(input_tensor, tuple):
            macs, params = profile(model, inputs=input_tensor, verbose=False)
        else:
            macs, params = profile(model, inputs=(input_tensor,), verbose=False)

        flops = 2 * macs  # Convert MACs to FLOPs (multiply-accumulate operations)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return {
            "flops": flops,
            "params": params,
            "flops_str": flops_str,
            "params_str": params_str,
            "macs": macs,
        }
    except Exception as e:
        print(f"  ⚠️  Error calculating FLOPs for {module_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_spformer_layers(model, input_shape, device):
    """Analyze SPFormer layer by layer following the actual forward pass"""
    print("=" * 100)
    print("SPFormer LAYER-BY-LAYER ANALYSIS")
    print("=" * 100)

    layer_results = {}

    print(f"Input shape for layer analysis: {input_shape}")

    # Create input tensor
    x = torch.randn(input_shape).to(device)
    original_input = x.clone()  # Keep original for spectral banks

    print("Starting Forward Pass Analysis:")
    print("-" * 50)

    # 1. PatchEmbed Layer
    print("Layer 1: PatchEmbed")
    print("-" * 30)

    # Store encoder outputs for skip connections
    encoder_outputs = {}

    # 1. PatchEmbed Layer
    print("Layer 1: PatchEmbed")
    print("-" * 30)

    try:
        patch_embed_result = count_module_flops(model.patch, x, "PatchEmbed")
        if patch_embed_result:
            print(f"   • Input: {x.shape}")
            x = model.patch(x)
            print(f"   • Output: {x.shape}")
            print(f"   • Parameters: {patch_embed_result['params_str']}")
            print(f"   • FLOPs: {patch_embed_result['flops_str']}")
            layer_results["patch_embed"] = patch_embed_result
    except Exception as e:
        print(f"     Error with PatchEmbed: {e}")

    # 2. Encoder Stage 1
    print(f"Layers 2-{1 + len(model.enc1)}: Encoder Stage 1 ({len(model.enc1)} blocks)")
    print("-" * 30)

    for i, block in enumerate(model.enc1):
        print(f"   Encoder1 Block {i + 1}:")

        try:
            block_name = f"enc1_block_{i}"
            block_result = count_module_flops(block, x, block_name)
            if block_result:
                print(f"      • Input: {x.shape}")
                x = block(x)
                print(f"      • Output: {x.shape}")
                print(f"      • Parameters: {block_result['params_str']}")
                print(f"      • FLOPs: {block_result['flops_str']}")
                layer_results[block_name] = block_result
        except Exception as e:
            print(f"      ⚠️  Error with Enc1 Block {i}: {e}")

    # Store e1 for skip connection
    encoder_outputs["e1"] = x.clone()

    # 3. Downsample 1
    print(f"Layer {2 + len(model.enc1)}: Downsample1")
    print("-" * 30)

    try:
        down1_result = count_module_flops(model.down1, x, "Downsample1")
        if down1_result:
            print(f"   • Input: {x.shape}")
            x = model.down1(x)
            print(f"   • Output: {x.shape}")
            print(f"   • Parameters: {down1_result['params_str']}")
            print(f"   • FLOPs: {down1_result['flops_str']}")
            layer_results["downsample1"] = down1_result
    except Exception as e:
        print(f"   ⚠️  Error with Downsample1: {e}")

    # 4. Encoder Stage 2
    print(
        f"Layers {3 + len(model.enc1)}-{2 + len(model.enc1) + len(model.enc2)}: Encoder Stage 2 ({len(model.enc2)} blocks)"
    )
    print("-" * 30)

    for i, block in enumerate(model.enc2):
        print(f"   Encoder2 Block {i + 1}:")

        try:
            block_name = f"enc2_block_{i}"
            block_result = count_module_flops(block, x, block_name)
            if block_result:
                print(f"      • Input: {x.shape}")
                x = block(x)
                print(f"      • Output: {x.shape}")
                print(f"      • Parameters: {block_result['params_str']}")
                print(f"      • FLOPs: {block_result['flops_str']}")
                layer_results[block_name] = block_result
        except Exception as e:
            print(f"      ⚠️  Error with Enc2 Block {i}: {e}")

    # Store e2 for skip connection
    encoder_outputs["e2"] = x.clone()

    # 5. Downsample 2
    print(f"Layer {3 + len(model.enc1) + len(model.enc2)}: Downsample2")
    print("-" * 30)

    try:
        down2_result = count_module_flops(model.down2, x, "Downsample2")
        if down2_result:
            print(f"   • Input: {x.shape}")
            x = model.down2(x)
            print(f"   • Output: {x.shape}")
            print(f"   • Parameters: {down2_result['params_str']}")
            print(f"   • FLOPs: {down2_result['flops_str']}")
            layer_results["downsample2"] = down2_result
    except Exception as e:
        print(f"   ⚠️  Error with Downsample2: {e}")

    # 6. Encoder Stage 3
    print(
        f"Layers {4 + len(model.enc1) + len(model.enc2)}-{3 + len(model.enc1) + len(model.enc2) + len(model.enc3)}: Encoder Stage 3 ({len(model.enc3)} blocks)"
    )
    print("-" * 30)

    for i, block in enumerate(model.enc3):
        print(f"   Encoder3 Block {i + 1}:")

        try:
            block_name = f"enc3_block_{i}"
            block_result = count_module_flops(block, x, block_name)
            if block_result:
                print(f"      • Input: {x.shape}")
                x = block(x)
                print(f"      • Output: {x.shape}")
                print(f"      • Parameters: {block_result['params_str']}")
                print(f"      • FLOPs: {block_result['flops_str']}")
                layer_results[block_name] = block_result
        except Exception as e:
            print(f"      ⚠️  Error with Enc3 Block {i}: {e}")

    # Store e3 for skip connection
    encoder_outputs["e3"] = x.clone()

    # 7. Downsample 3
    print(
        f"Layer {4 + len(model.enc1) + len(model.enc2) + len(model.enc3)}: Downsample3"
    )
    print("-" * 30)

    try:
        down3_result = count_module_flops(model.down3, x, "Downsample3")
        if down3_result:
            print(f"   • Input: {x.shape}")
            x = model.down3(x)
            print(f"   • Output: {x.shape}")
            print(f"   • Parameters: {down3_result['params_str']}")
            print(f"   • FLOPs: {down3_result['flops_str']}")
            layer_results["downsample3"] = down3_result
    except Exception as e:
        print(f"   ⚠️  Error with Downsample3: {e}")

    # 8. Latent Blocks
    print(
        f"Layers {5 + len(model.enc1) + len(model.enc2) + len(model.enc3)}-{4 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks)}: Latent Blocks ({len(model.latent_blocks)} blocks)"
    )
    print("-" * 30)

    for i, block in enumerate(model.latent_blocks):
        print(f"   Latent Block {i + 1}:")

        try:
            block_name = f"latent_block_{i}"
            block_result = count_module_flops(block, x, block_name)
            if block_result:
                print(f"      • Input: {x.shape}")
                x = block(x)
                print(f"      • Output: {x.shape}")
                print(f"      • Parameters: {block_result['params_str']}")
                print(f"      • FLOPs: {block_result['flops_str']}")
                layer_results[block_name] = block_result
        except Exception as e:
            print(f"      ⚠️  Error with Latent Block {i}: {e}")

    # 9. SpectralBank at Latent Level
    print(
        f"Layer {5 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks)}: SpectralBank Latent"
    )
    print("-" * 30)

    try:
        sbm_latent_result = count_module_flops(
            model.sbm_latent, (x, original_input), "SpectralBank_Latent"
        )
        if sbm_latent_result:
            print(f"   • Feature input: {x.shape}")
            print(f"   • Guidance input: {original_input.shape}")
            sbm_lat_out = model.sbm_latent(x, original_input)
            print(f"   • Output: {sbm_lat_out.shape}")
            print(f"   • Parameters: {sbm_latent_result['params_str']}")
            print(f"   • FLOPs: {sbm_latent_result['flops_str']}")
            layer_results["spectral_bank_latent"] = sbm_latent_result

            # Add residual connection (latent + sbm_latent)
            x = x + sbm_lat_out
            print(f"   • After residual: {x.shape}")
    except Exception as e:
        print(f"   ⚠️  Error with SpectralBank Latent: {e}")

    print("Starting Decoder Path:")
    print("-" * 30)

    # 10. Upsample 3 + Skip Connection
    print(
        f"📊 Layer {6 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks)}: Upsample3 + Skip"
    )
    print("-" * 30)

    try:
        up3_result = count_module_flops(model.up3, x, "Upsample3")
        if up3_result:
            print(f"   • Input: {x.shape}")
            x = model.up3(x)
            print(f"   • After upsample: {x.shape}")

            # Skip connection with e3
            e3 = encoder_outputs["e3"]
            print(f"   • Skip connection with e3: {e3.shape}")
            x = x + e3
            print(f"   • After skip connection: {x.shape}")

            print(f"   • Parameters: {up3_result['params_str']}")
            print(f"   • FLOPs: {up3_result['flops_str']}")
            layer_results["upsample3"] = up3_result
    except Exception as e:
        print(f"   ⚠️  Error with Upsample3: {e}")

    # 11. Decoder Stage 3
    print(
        f"📊 Layers {7 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks)}-{6 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3)}: Decoder Stage 3 ({len(model.dec3)} blocks)"
    )
    print("-" * 30)

    for i, block in enumerate(model.dec3):
        print(f"   🔍 Decoder3 Block {i + 1}:")

        try:
            block_name = f"dec3_block_{i}"
            block_result = count_module_flops(block, x, block_name)
            if block_result:
                print(f"      • Input: {x.shape}")
                x = block(x)
                print(f"      • Output: {x.shape}")
                print(f"      • Parameters: {block_result['params_str']}")
                print(f"      • FLOPs: {block_result['flops_str']}")
                layer_results[block_name] = block_result
        except Exception as e:
            print(f"      ⚠️  Error with Dec3 Block {i}: {e}")

    # 12. SpectralBank3
    print(
        f"📊 Layer {7 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3)}: SpectralBank3"
    )
    print("-" * 30)

    try:
        sbm3_result = count_module_flops(
            model.sbm3, (x, original_input), "SpectralBank3"
        )
        if sbm3_result:
            print(f"   • Feature input: {x.shape}")
            print(f"   • Guidance input: {original_input.shape}")
            sbm3_out = model.sbm3(x, original_input)
            print(f"   • Output: {sbm3_out.shape}")
            print(f"   • Parameters: {sbm3_result['params_str']}")
            print(f"   • FLOPs: {sbm3_result['flops_str']}")
            layer_results["spectral_bank3"] = sbm3_result

            # Add residual connection (d3r + sbm3)
            x = x + sbm3_out
            print(f"   • After residual: {x.shape}")
    except Exception as e:
        print(f"   ⚠️  Error with SpectralBank3: {e}")

    # 13. Upsample 2 + Skip Connection
    print(
        f"📊 Layer {8 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3)}: Upsample2 + Skip"
    )
    print("-" * 30)

    try:
        up2_result = count_module_flops(model.up2, x, "Upsample2")
        if up2_result:
            print(f"   • Input: {x.shape}")
            x = model.up2(x)
            print(f"   • After upsample: {x.shape}")

            # Skip connection with e2
            e2 = encoder_outputs["e2"]
            print(f"   • Skip connection with e2: {e2.shape}")
            x = x + e2
            print(f"   • After skip connection: {x.shape}")

            print(f"   • Parameters: {up2_result['params_str']}")
            print(f"   • FLOPs: {up2_result['flops_str']}")
            layer_results["upsample2"] = up2_result
    except Exception as e:
        print(f"   ⚠️  Error with Upsample2: {e}")

    # 14. Decoder Stage 2
    print(
        f"📊 Layers {9 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3)}-{8 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3) + len(model.dec2)}: Decoder Stage 2 ({len(model.dec2)} blocks)"
    )
    print("-" * 30)

    for i, block in enumerate(model.dec2):
        print(f"   🔍 Decoder2 Block {i + 1}:")

        try:
            block_name = f"dec2_block_{i}"
            block_result = count_module_flops(block, x, block_name)
            if block_result:
                print(f"      • Input: {x.shape}")
                x = block(x)
                print(f"      • Output: {x.shape}")
                print(f"      • Parameters: {block_result['params_str']}")
                print(f"      • FLOPs: {block_result['flops_str']}")
                layer_results[block_name] = block_result
        except Exception as e:
            print(f"      ⚠️  Error with Dec2 Block {i}: {e}")

    # 15. SpectralBank2
    print(
        f"📊 Layer {9 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3) + len(model.dec2)}: SpectralBank2"
    )
    print("-" * 30)

    try:
        sbm2_result = count_module_flops(
            model.sbm2, (x, original_input), "SpectralBank2"
        )
        if sbm2_result:
            print(f"   • Feature input: {x.shape}")
            print(f"   • Guidance input: {original_input.shape}")
            sbm2_out = model.sbm2(x, original_input)
            print(f"   • Output: {sbm2_out.shape}")
            print(f"   • Parameters: {sbm2_result['params_str']}")
            print(f"   • FLOPs: {sbm2_result['flops_str']}")
            layer_results["spectral_bank2"] = sbm2_result

            # Add residual connection (d2r + sbm2)
            x = x + sbm2_out
            print(f"   • After residual: {x.shape}")
    except Exception as e:
        print(f"   ⚠️  Error with SpectralBank2: {e}")

    # 16. Upsample 1 + Skip Connection
    print(
        f"📊 Layer {10 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3) + len(model.dec2)}: Upsample1 + Skip"
    )
    print("-" * 30)

    try:
        up1_result = count_module_flops(model.up1, x, "Upsample1")
        if up1_result:
            print(f"   • Input: {x.shape}")
            x = model.up1(x)
            print(f"   • After upsample: {x.shape}")

            # Skip connection with e1
            e1 = encoder_outputs["e1"]
            print(f"   • Skip connection with e1: {e1.shape}")
            x = x + e1
            print(f"   • After skip connection: {x.shape}")

            print(f"   • Parameters: {up1_result['params_str']}")
            print(f"   • FLOPs: {up1_result['flops_str']}")
            layer_results["upsample1"] = up1_result
    except Exception as e:
        print(f"   ⚠️  Error with Upsample1: {e}")

    # 17. Decoder Stage 1
    print(
        f"📊 Layers {11 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3) + len(model.dec2)}-{10 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3) + len(model.dec2) + len(model.dec1)}: Decoder Stage 1 ({len(model.dec1)} blocks)"
    )
    print("-" * 30)

    for i, block in enumerate(model.dec1):
        print(f"   🔍 Decoder1 Block {i + 1}:")

        try:
            block_name = f"dec1_block_{i}"
            block_result = count_module_flops(block, x, block_name)
            if block_result:
                print(f"      • Input: {x.shape}")
                x = block(x)
                print(f"      • Output: {x.shape}")
                print(f"      • Parameters: {block_result['params_str']}")
                print(f"      • FLOPs: {block_result['flops_str']}")
                layer_results[block_name] = block_result
        except Exception as e:
            print(f"      ⚠️  Error with Dec1 Block {i}: {e}")

    # 18. SpectralBank1
    print(
        f"📊 Layer {11 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3) + len(model.dec2) + len(model.dec1)}: SpectralBank1"
    )
    print("-" * 30)

    try:
        sbm1_result = count_module_flops(
            model.sbm1, (x, original_input), "SpectralBank1"
        )
        if sbm1_result:
            print(f"   • Feature input: {x.shape}")
            print(f"   • Guidance input: {original_input.shape}")
            sbm1_out = model.sbm1(x, original_input)
            print(f"   • Output: {sbm1_out.shape}")
            print(f"   • Parameters: {sbm1_result['params_str']}")
            print(f"   • FLOPs: {sbm1_result['flops_str']}")
            layer_results["spectral_bank1"] = sbm1_result

            # Add residual connection (d1r + sbm1)
            x = x + sbm1_out
            print(f"   • After residual: {x.shape}")
    except Exception as e:
        print(f"   ⚠️  Error with SpectralBank1: {e}")

    # 19. Final Convolution
    print(
        f"📊 Layer {12 + len(model.enc1) + len(model.enc2) + len(model.enc3) + len(model.latent_blocks) + len(model.dec3) + len(model.dec2) + len(model.dec1)}: Final Convolution"
    )
    print("-" * 30)

    try:
        final_conv_result = count_module_flops(model.final_conv, x, "Final_Conv")
        if final_conv_result:
            print(f"   • Input: {x.shape}")
            x = model.final_conv(x)
            print(f"   • Output: {x.shape}")
            print(f"   • Parameters: {final_conv_result['params_str']}")
            print(f"   • FLOPs: {final_conv_result['flops_str']}")
            layer_results["final_conv"] = final_conv_result

            # Final residual with original input
            print(f"   • Adding residual with original input: {original_input.shape}")
            x = torch.clamp(x + original_input, 0.0, 1.0)
            print(f"   • Final output: {x.shape}")
    except Exception as e:
        print(f"   ⚠️  Error with Final Conv: {e}")

    print("Complete layer-by-layer analysis finished!")
    return layer_results


def main(args, cfg):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_data = cfg.get("data", {})
    cfg_model = cfg.get("model", {})

    # Override image size if provided (command line takes priority)
    image_size = args.image_size
    if image_size == 256 and "image_size" in cfg_data:
        # Only use config image_size if command line is default (256)
        image_size = cfg_data.get("image_size", image_size)
    print(
        f"Using image size: {image_size} (command line: {args.image_size}, config: {cfg_data.get('image_size', 'not set')})"
    )

    batch_size = args.batch_size
    input_shape = (batch_size, 3, image_size, image_size)

    print(f"Input shape for FLOPs calculation: {input_shape}")

    try:
        # Create SPFormer model with new architecture
        # Use default parameters if not specified in config
        base_ch = cfg_model.get("base_ch", 20)  # Default from net.py
        blocks = cfg_model.get("blocks", [2, 2, 2, 2])  # Default from net.py
        heads = cfg_model.get("heads", [2, 4, 8, 8])  # Default from net.py
        ffn_expansion = cfg_model.get("ffn_expansion", 1.5)  # Default from net.py
        bands = cfg_model.get("bands", 4)  # Default from net.py

        model = SPFormer(
            base_ch=base_ch,
            blocks=blocks,
            heads=heads,
            ffn_expansion=ffn_expansion,
            bands=bands,
        )
        model = model.to(device)

        # Store config for later use in analysis
        model.model_cfg = cfg_model

        n_parameters = utils.count_model_parameters(model)
        print(
            f"SPFormer (New Architecture) created successfully with {n_parameters:,} parameters"
        )

        # Print model architecture summary
        print("Model Architecture Summary (New):")
        print(f"  • Base channels: {base_ch}")
        print(f"  • Encoder blocks: {blocks}")
        print(f"  • Attention heads: {heads}")
        print(f"  • FFN Expansion: {ffn_expansion}")
        print(f"  • Spectral bands: {bands}")

        # Show channel dimensions for each stage
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        print(f"  • Stage dimensions: C1={c1}, C2={c2}, C3={c3}, C4={c4}")

    except Exception as e:
        print(f"Error creating SPFormer model: {e}")
        print("Model config:", cfg_model)
        raise

    # Analyze layers following forward pass
    layer_results = analyze_spformer_layers(model, input_shape, device)

    # Summary
    print("" + "=" * 100)
    print("SUMMARY - SPFormer LAYER-BY-LAYER FLOPs BREAKDOWN")
    print("=" * 100)

    total_flops = 0
    total_params = 0

    for layer_name, result in layer_results.items():
        if result:
            flops_val = result["flops"]
            params_val = result["params"]
            total_flops += flops_val
            total_params += params_val

    # Print detailed breakdown
    print("DETAILED LAYER ANALYSIS:")
    print("-" * 100)
    print(f"{'Layer':<25} {'FLOPs':<12} {'Params':<12} {'FLOPs%':<8} {'Params%':<8}")
    print("-" * 100)

    for layer_name, result in layer_results.items():
        if result:
            flops_val = result["flops"]
            params_val = result["params"]
            flops_percentage = flops_val / total_flops if total_flops > 0 else 0
            params_percentage = params_val / total_params if total_params > 0 else 0
            print(
                f"{layer_name:<25} {result['flops_str']:>12} {result['params_str']:>12} "
                f"{flops_percentage:>6.1%} {params_percentage:>6.1%}"
            )

    print("-" * 100)
    total_flops_str, total_params_str = clever_format(
        [total_flops, total_params], "%.3f"
    )
    print(
        f"{'TOTAL LAYERS':<25} {total_flops_str:>12} {total_params_str:>12} {'100.0%':>8} {'100.0%':>8}"
    )

    # Compare with full model
    print("" + "=" * 50)
    print("FULL MODEL VERIFICATION")
    print("=" * 50)

    x = torch.randn(input_shape).to(device)
    full_model_result = count_module_flops(model, x, "Full SPFormer")

    if full_model_result:
        print(f"Full model FLOPs: {full_model_result['flops_str']}")
        print(f"Full model params: {full_model_result['params_str']}")

        if total_flops > 0:
            ratio = full_model_result["flops"] / total_flops
            print(f"Full model / Layer sum ratio: {ratio:.2f}")
            if abs(ratio - 1.0) > 0.1:  # More than 10% difference
                print("⚠️  Significant difference - check layer analysis")
                print("   This might be due to residual connections or skipped layers")

    # Layer-wise performance insights
    print("" + "=" * 50)
    print("LAYER PERFORMANCE INSIGHTS")
    print("=" * 50)

    # Find most expensive layers
    if layer_results:
        sorted_layers = sorted(
            layer_results.items(), key=lambda x: x[1]["flops"], reverse=True
        )

        print("🔥 TOP 5 MOST EXPENSIVE LAYERS:")
        print("-" * 60)
        for i, (layer_name, result) in enumerate(sorted_layers[:5]):
            flops_percentage = result["flops"] / total_flops if total_flops > 0 else 0
            print(
                f"{i + 1}. {layer_name:<20} {result['flops_str']:>12} ({flops_percentage:.1%})"
            )

        # Encoder vs Decoder analysis
        encoder_flops = sum(
            result["flops"]
            for name, result in layer_results.items()
            if "encoder" in name or "latent" in name
        )
        spectral_flops = sum(
            result["flops"]
            for name, result in layer_results.items()
            if "spectral" in name
        )
        down_up_flops = sum(
            result["flops"]
            for name, result in layer_results.items()
            if "down" in name or "up" in name or "patch" in name
        )

        print("COMPUTATION DISTRIBUTION:")
        print("-" * 60)
        if total_flops > 0:
            encoder_percentage = encoder_flops / total_flops * 100
            spectral_percentage = spectral_flops / total_flops * 100
            down_up_percentage = down_up_flops / total_flops * 100

            print(
                f"Encoder/Decoder/Latent: {encoder_flops:,} ({encoder_percentage:.1f}%)"
            )
            print(f"Spectral Banks: {spectral_flops:,} ({spectral_percentage:.1f}%)")
            print(f"Down/Up/Conv: {down_up_flops:,} ({down_up_percentage:.1f}%)")

            # More detailed breakdown
            print("DETAILED BREAKDOWN:")
            print("-" * 40)

            # Encoder stages breakdown
            enc1_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "enc1" in name
            )
            enc2_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "enc2" in name
            )
            enc3_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "enc3" in name
            )
            latent_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "latent" in name
            )
            dec1_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "dec1" in name
            )
            dec2_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "dec2" in name
            )
            dec3_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "dec3" in name
            )

            if enc1_flops > 0:
                print(f"  Enc1: {enc1_flops:,} ({enc1_flops / total_flops * 100:.1f}%)")
            if enc2_flops > 0:
                print(f"  Enc2: {enc2_flops:,} ({enc2_flops / total_flops * 100:.1f}%)")
            if enc3_flops > 0:
                print(f"  Enc3: {enc3_flops:,} ({enc3_flops / total_flops * 100:.1f}%)")
            if latent_flops > 0:
                print(
                    f"  Latent: {latent_flops:,} ({latent_flops / total_flops * 100:.1f}%)"
                )
            if dec1_flops > 0:
                print(f"  Dec1: {dec1_flops:,} ({dec1_flops / total_flops * 100:.1f}%)")
            if dec2_flops > 0:
                print(f"  Dec2: {dec2_flops:,} ({dec2_flops / total_flops * 100:.1f}%)")
            if dec3_flops > 0:
                print(f"  Dec3: {dec3_flops:,} ({dec3_flops / total_flops * 100:.1f}%)")

            # Spectral banks breakdown
            sb_latent_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "spectral_bank_latent" in name
            )
            sb3_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "spectral_bank3" in name
            )
            sb2_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "spectral_bank2" in name
            )
            sb1_flops = sum(
                result["flops"]
                for name, result in layer_results.items()
                if "spectral_bank1" in name
            )

            if sb_latent_flops > 0:
                print(
                    f"  SB_Latent: {sb_latent_flops:,} ({sb_latent_flops / total_flops * 100:.1f}%)"
                )
            if sb3_flops > 0:
                print(f"  SB3: {sb3_flops:,} ({sb3_flops / total_flops * 100:.1f}%)")
            if sb2_flops > 0:
                print(f"  SB2: {sb2_flops:,} ({sb2_flops / total_flops * 100:.1f}%)")
            if sb1_flops > 0:
                print(f"  SB1: {sb1_flops:,} ({sb1_flops / total_flops * 100:.1f}%)")

    print("ARCHITECTURE EFFICIENCY (NEW):")
    print("-" * 60)

    # Get config for new architecture
    base_ch = cfg_model.get("base_ch", 20)
    blocks = cfg_model.get("blocks", [2, 2, 2, 2])
    bands = cfg_model.get("bands", 4)

    total_transformer_blocks = sum(blocks)
    total_spectral_banks = 4  # sbm_latent + sbm3 + sbm2 + sbm1

    print(f"  • Total transformer blocks: {total_transformer_blocks}")
    print(f"  • Total spectral banks: {total_spectral_banks}")
    print(f"  • Total layers analyzed: {len(layer_results)}")
    print(
        f"  • Average FLOPs per layer: {total_flops / len(layer_results):.2e}"
        if layer_results
        else "  • No layers analyzed"
    )

    if full_model_result:
        params_per_flop = full_model_result["params"] / full_model_result["flops"]
        print(f"  • Parameters per FLOP: {params_per_flop:.2f}")
        print(
            f"  • Model efficiency: {'High' if params_per_flop < 1.0 else 'Medium' if params_per_flop < 2.0 else 'Low'}"
        )

    # New architecture specific analysis
    print("ARCHITECTURE SPECIFICS:")
    print(f"  • Base channels: {base_ch} → Final: {base_ch * 8}")
    print("  • Progressive upsampling: 4 stages")
    print("  • Skip connections: 3 levels (e1, e2, e3)")
    print("  • Spectral enhancement: 4 levels")
    print("  • Final residual: Output + Input")

    print("" + "=" * 100)
    print("✅ SPFormer FLOPs ANALYSIS COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        "SPFormer FLOPs Calculator", parents=[get_args_parser()]
    )
    parser.add_argument("--cfg", type=str, default="configs/lsui.yaml")
    args = parser.parse_args()

    try:
        with open(args.cfg, "r+", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        Path(config.get("training", {}).get("model_dir", "models")).mkdir(
            parents=True, exist_ok=True
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file {args.cfg} not found, using default SPFormer configuration"
        )

    main(args, config)
