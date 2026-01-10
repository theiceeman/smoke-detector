"""Unified training functions for all Ultralytics models"""

from pathlib import Path
from models.unified import create_model


def train_model(
    model_name: str,
    dataset_yaml: str,
    num_epochs: int = 20,
    patience: int = 6,
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
    
    # Train model
    # Note: Ultralytics saves to runs/detect/{project}/{name}/weights/best.pt
    print(f"  Starting training for {model_name.upper()}...")
    results = model.train(
        data=dataset_yaml,
        epochs=num_epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        project=save_dir,  # Creates runs/detect/{save_dir}/
        name=model_name,    # Creates runs/detect/{save_dir}/{model_name}/
        save=True,
        plots=True,
        verbose=True
    )
    
    # Path to best model (Ultralytics structure: runs/detect/{project}/{name}/weights/best.pt)
    best_model_path = Path("runs") / "detect" / save_dir / model_name / "weights" / "best.pt"
    
    # If not found, try to get from results object
    if not best_model_path.exists():
        if hasattr(results, 'save_dir'):
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    return str(best_model_path)


