import random
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def get_image_files(folder: Path) -> List[Path]:
    """Get all image files in folder"""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []

    return sorted(
        [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )


def find_matching_pairs(dir1: Path, dir2: Path) -> List[Tuple[Path, Path]]:
    """Find matching pairs of files with the same name (without extension)"""
    files1 = get_image_files(dir1)
    files2 = get_image_files(dir2)

    # Create a dict mapping file name (without extension) to path
    files2_dict = {f.stem: f for f in files2}

    pairs = []
    for f1 in files1:
        if f1.stem in files2_dict:
            pairs.append((f1, files2_dict[f1.stem]))

    return pairs


def detect_best_subdirs(dataset_path: Path) -> Tuple[str, str]:
    """Automatically detect the best 2 subdirs with the most matching files"""
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if len(subdirs) < 2:
        raise ValueError(
            f"Dataset must have at least 2 subdirectories. Found: {len(subdirs)}"
        )

    best_pair = None
    max_matches = 0

    # Try all pairs of subdirs
    for i in range(len(subdirs)):
        for j in range(i + 1, len(subdirs)):
            dir1, dir2 = subdirs[i], subdirs[j]
            pairs = find_matching_pairs(dir1, dir2)

            if len(pairs) > max_matches:
                max_matches = len(pairs)
                best_pair = (dir1.name, dir2.name)

    if max_matches == 0:
        raise ValueError("No matching files found between subdirectories")

    print(
        f"Detected best pair: '{best_pair[0]}' and '{best_pair[1]}' ({max_matches} pairs)"
    )
    return best_pair


def split_dataset(dataset_path: Path, test_ratio: float = 0.2, seed: int = 42):
    """Split dataset into train/test"""

    # Automatically detect 2 subdirs
    subdir1_name, subdir2_name = detect_best_subdirs(dataset_path)

    subdir1 = dataset_path / subdir1_name
    subdir2 = dataset_path / subdir2_name

    # Find matching pairs of files
    pairs = find_matching_pairs(subdir1, subdir2)
    print(f"Found {len(pairs)} matching pairs")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)

    split_point = int(len(pairs) * (1 - test_ratio))
    train_pairs = pairs[:split_point]
    test_pairs = pairs[split_point:]

    print(f"Train: {len(train_pairs)} pairs, Test: {len(test_pairs)} pairs")

    # Create directory structure
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    # Delete and recreate if they exist
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Create subdirs in train and test
    (train_dir / subdir1_name).mkdir(parents=True)
    (train_dir / subdir2_name).mkdir(parents=True)
    (test_dir / subdir1_name).mkdir(parents=True)
    (test_dir / subdir2_name).mkdir(parents=True)

    # Copy files to train and test
    print("Copying files...")

    for file1, file2 in tqdm(train_pairs, desc="Copying train pairs", unit="pair"):
        shutil.copy2(file1, train_dir / subdir1_name / file1.name)
        shutil.copy2(file2, train_dir / subdir2_name / file2.name)

    for file1, file2 in tqdm(test_pairs, desc="Copying test pairs", unit="pair"):
        shutil.copy2(file1, test_dir / subdir1_name / file1.name)
        shutil.copy2(file2, test_dir / subdir2_name / file2.name)

    print("Done!")
    print(f"Train set: {train_dir}")
    print(f"Test set: {test_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/test")
    parser.add_argument("dataset_path", type=str, help="Path to dataset folder")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio (0.0-1.0), default 0.2 (20%%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    if not (0.0 <= args.test_ratio <= 1.0):
        print("Error: test-ratio must be between 0.0 and 1.0")
        return

    try:
        split_dataset(dataset_path, args.test_ratio, args.seed)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
