import os
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
import cv2
import math
from torchvision.utils import make_grid


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(
                f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}"
            )
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag="color", float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError("Failed in writing images.")


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]


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

        combined = torch.cat([input_img, pred_img, target_img], dim=2)
        combined_path = os.path.join(save_dir, f"{filename}.png")
        save_img(combined, combined_path)
