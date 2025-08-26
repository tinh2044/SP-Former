import os
from pathlib import Path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class UIEDataset(data.Dataset):
    def __init__(self, root, cfg, split="train"):
        self.root = root
        self.cfg = cfg
        self.split = split

        # Setup paths
        if split == "train":
            self.input_dir = os.path.join(root, cfg["train_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["train_dir"], cfg["target_dir"])
        else:
            self.input_dir = os.path.join(root, cfg["test_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["test_dir"], cfg["target_dir"])

        # Get image files
        self.image_files = self._get_image_files()
        self.extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        # Setup transforms
        self.transform = self._get_transforms()

        print(f"UIEDataset[{split}] -> {len(self.image_files)} samples")
        print(
            f"  Input: {self.input_dir} ({len(self._list_files(self.input_dir))} files)"
        )
        print(
            f"  Target: {self.target_dir} ({len(self._list_files(self.target_dir))} files)"
        )

    def _list_files(self, folder):
        if not os.path.exists(folder):
            return []
        files = []
        for fp in Path(folder).rglob("*"):
            if fp.is_file() and fp.suffix.lower() in self.extensions:
                files.append(fp)
        return files

    def _get_image_files(self):
        """Get list of image files"""
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = []

        if os.path.exists(self.input_dir) and os.path.exists(self.target_dir):
            input_files = sorted(
                [
                    f
                    for f in os.listdir(self.input_dir)
                    if f.lower().endswith(valid_extensions)
                ]
            )

            for file in input_files:
                target_file = os.path.join(self.target_dir, file)
                if os.path.exists(target_file):
                    image_files.append(file)

        return image_files

    def _get_transforms(self):
        """Get image transforms"""
        image_size = self.cfg.get("image_size", 256)

        if self.split == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        return transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get a single sample"""
        filename = self.image_files[idx]

        # Load images
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Apply transforms
        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": filename,
            "idx": idx,
        }

    def data_collator(self, batch):
        """Custom collate function for batching"""
        inputs = torch.stack([item["input"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        filenames = [item["filename"] for item in batch]
        indices = [item["idx"] for item in batch]

        return {
            "inputs": inputs,
            "targets": targets,
            "filenames": filenames,
            "indices": indices,
        }


def get_training_set(root, cfg):
    """Get training dataset"""
    return UIEDataset(root, cfg, split="train")


def get_test_set(root, cfg):
    """Get test dataset"""
    return UIEDataset(root, cfg, split="test")
