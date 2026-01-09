"""Data loading utilities for smoke detection experiment"""

from pathlib import Path
from typing import List


def load_test_images(test_dir: str) -> List[Path]:
    """
    Load all test images from directory.
    
    Args:
        test_dir: Directory containing test images
    
    Returns:
        List of image paths
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Load all image files
    images = (
        list(test_path.glob("*.jpg")) +
        list(test_path.glob("*.png")) +
        list(test_path.glob("*.jpeg"))
    )
    
    return sorted(images)


def prepare_yolo_dataset(data_root: str) -> str:
    """
    Prepare YOLO dataset (returns path to data.yaml).
    
    Args:
        data_root: Root directory containing dataset
    
    Returns:
        Path to data.yaml file
    """
    data_root = Path(data_root)
    dataset_yaml = data_root / "data.yaml"
    
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    return str(dataset_yaml)


