"""Unified model creation for all Ultralytics models (YOLOv8, YOLOv11, RT-DETR)"""

from pathlib import Path


def create_model(model_name: str, pretrained: bool = True, model_path: str = None):
    """
    Create any Ultralytics model (YOLOv8, YOLOv11, RT-DETR).
    
    Args:
        model_name: 'yolov8', 'yolov11', or 'rtdetr'
        pretrained: If True, load pre-trained weights
        model_path: Path to trained model weights (if not pretrained)
    
    Returns:
        Model instance
    """
    if model_path and Path(model_path).exists():
        # Load trained model - try to auto-detect type
        return _load_trained_model(model_path)
    
    if not pretrained:
        raise ValueError("Either pretrained=True or model_path must be provided")
    
    # Load pre-trained model
    if model_name.lower() == 'yolov8':
        from ultralytics import YOLO
        return YOLO('yolov8n.pt')
    
    elif model_name.lower() == 'yolov11':
        from ultralytics import YOLO
        try:
            return YOLO('yolov11n.pt')
        except:
            # Fallback to YOLOv8 if v11 not available
            print("  ⚠ YOLOv11 not available, using YOLOv8")
            return YOLO('yolov8n.pt')
    
    elif model_name.lower() == 'rtdetr':
        try:
            from ultralytics import RTDETR
            return RTDETR('rtdetr-l.pt')
        except ImportError:
            raise ImportError("RT-DETR not available. Install ultralytics with RT-DETR support.")
        except Exception as e:
            print(f"  ⚠ RT-DETR loading failed: {e}")
            raise
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. Use 'yolov8', 'yolov11', or 'rtdetr'")


def _load_trained_model(model_path: str):
    """
    Load trained model - auto-detect type from path or try loading.
    
    Args:
        model_path: Path to trained model weights
    
    Returns:
        Model instance
    """
    model_path = Path(model_path)
    
    # Try YOLO first (works for YOLOv8 and YOLOv11)
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        # Test if it loads correctly
        _ = model.model  # Try to access model attribute
        return model
    except:
        pass
    
    # Try RT-DETR
    try:
        from ultralytics import RTDETR
        model = RTDETR(str(model_path))
        _ = model.model
        return model
    except:
        pass
    
    # If both fail, raise error
    raise ValueError(f"Could not load model from {model_path}. Unknown model type.")


