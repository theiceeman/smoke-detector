"""Unified training functions for all Ultralytics models"""

from pathlib import Path
from models.unified import create_model


def train_model(
    model_name: str,
    dataset_yaml: str,
    num_epochs: int = 50,
    patience: int = 10,
    imgsz: int = 640,
    batch: int = 16,
    save_dir: str = "checkpoints"
):
    """
    Train any Ultralytics model (YOLOv8, YOLOv11, RT-DETR).
    
    Args:
        model_name: 'yolov8', 'yolov11', or 'rtdetr'
        dataset_yaml: Path to dataset YAML file
        num_epochs: Number of training epochs
        patience: Early stopping patience
        imgsz: Image size for training
        batch: Batch size
        save_dir: Directory to save checkpoints
    
    Returns:
        Path to best model weights
    """
    # Create pre-trained model
    model = create_model(model_name, pretrained=True)
    
    # Setup save directory
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Train model
    results = model.train(
        data=dataset_yaml,
        epochs=num_epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        project=str(save_path),
        name="train",
        save=True,
        plots=True,
        verbose=True
    )
    
    # Path to best model
    best_model_path = save_path / "train" / "weights" / "best.pt"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    return str(best_model_path)


