from .img_util import crop_border, imfrombytes, img2tensor, imwrite, tensor2img
from .logger import (
    AvgTimer,
    MessageLogger,
    get_env_info,
    get_root_logger,
    init_tb_logger,
    init_wandb_logger,
)

__all__ = [
    # img_util.py
    "img2tensor",
    "tensor2img",
    "imfrombytes",
    "imwrite",
    "crop_border",
    # logger.py
    "MessageLogger",
    "AvgTimer",
    "init_tb_logger",
    "init_wandb_logger",
    "get_root_logger",
    "get_env_info",
]
import os
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image


def save_img(image_tensor, filename):
    """Save image tensor to file"""
    image_numpy = image_tensor.detach().float().cpu().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)


def count_model_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_shape, device="cpu"):
    """Calculate FLOPs for model"""
    try:
        from thop import profile

        input_tensor = torch.randn(input_shape).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params
    except ImportError:
        print("thop not installed, skipping FLOPs calculation")
        return None, None


def get_model_info(model, input_shape, device="cpu"):
    """Get comprehensive model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }

    flops, params = calculate_flops(model, input_shape, device)
    if flops is not None:
        info["flops"] = flops
        info["flops_str"] = f"{flops / 1e9:.2f}G"
        info["macs"] = flops / 2
        info["macs_str"] = f"{flops / 2e9:.2f}G"
        info["params_str"] = f"{params / 1e6:.2f}M"

    return info


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get world size for distributed training"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank for distributed training"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if current process is main process"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save only on main process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """Setup for distributed training"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def check_state_dict(model, state_dict):
    """Check if model and state dict are compatible"""
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())

    # Check if all model keys are in state dict
    missing_keys = model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state dict: {missing_keys}")
        return False

    # Check if all state dict keys are in model
    unexpected_keys = state_dict_keys - model_keys
    if unexpected_keys:
        print(f"Unexpected keys in state dict: {unexpected_keys}")
        return False

    return True


def save_checkpoints(model, optimizer, scheduler, epoch, loss, path_dir, name=None):
    """Save training checkpoints"""
    os.makedirs(path_dir, exist_ok=True)

    if name is None:
        name = f"checkpoint_epoch_{epoch}.pth"

    checkpoint_path = os.path.join(path_dir, name)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_checkpoints(model, optimizer, scheduler, path, resume=True):
    """Load training checkpoints"""
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return 0, 0.0

    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if resume and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if resume and scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")

    return epoch, loss


def save_sample_images(inputs, pred, targets, batch_idx, epoch, output_dir):
    """Save sample images during training"""
    os.makedirs(output_dir, exist_ok=True)

    # Save first image in batch
    input_img = inputs[0]
    pred_img = pred[0]
    target_img = targets[0]

    # Concatenate images horizontally
    combined = torch.cat([input_img, pred_img, target_img], dim=2)

    filename = os.path.join(output_dir, f"sample_{batch_idx}_{epoch}.png")
    save_img(combined, filename)


def save_eval_images(inputs, pred, targets, filenames, epoch, output_dir):
    """Save evaluation images"""
    save_dir = os.path.join(output_dir, "eval", f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    for i, filename in enumerate(filenames):
        input_img = inputs[i]
        pred_img = pred[i]
        target_img = targets[i]

        # Save individual images
        input_path = os.path.join(save_dir, f"{filename}_input.png")
        pred_path = os.path.join(save_dir, f"{filename}_pred.png")
        target_path = os.path.join(save_dir, f"{filename}_target.png")

        save_img(input_img, input_path)
        save_img(pred_img, pred_path)
        save_img(target_img, target_path)

        # Save combined image
        combined = torch.cat([input_img, pred_img, target_img], dim=2)
        combined_path = os.path.join(save_dir, f"{filename}_combined.png")
        save_img(combined, combined_path)
