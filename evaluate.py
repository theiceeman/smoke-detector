"""Unified evaluation functions for all Ultralytics models"""

from pathlib import Path
from typing import List, Dict
import yaml
from models.unified import create_model


def evaluate_baseline(model, test_images: List[Path]) -> Dict:
    """
    Evaluate pre-trained model on test set (baseline).
    
    Args:
        model: Pre-trained model instance
        test_images: List of test image paths
    
    Returns:
        Dictionary with baseline metrics
    """
    images_with_detections = 0
    total_detections = 0
    detection_details = []
    
    for img_path in test_images:
        results = model(str(img_path), verbose=False)
        result = results[0]
        num_detections = len(result.boxes)
        
        if num_detections > 0:
            images_with_detections += 1
            total_detections += num_detections
        
        detection_details.append({
            'image': img_path.name,
            'detections': num_detections,
            'detected': num_detections > 0
        })
    
    return {
        'images_with_detections': images_with_detections,
        'total_detections': total_detections,
        'total_images': len(test_images),
        'detection_rate': images_with_detections / len(test_images) if test_images else 0,
        'details': detection_details
    }


def evaluate_trained(
    model_path: str,
    dataset_yaml: str,
    test_images: List[Path]
) -> Dict:
    """
    Evaluate trained model on test set with proper ground truth comparison.
    
    Args:
        model_path: Path to trained model weights
        dataset_yaml: Path to dataset YAML file
        test_images: List of test image paths
    
    Returns:
        Dictionary with trained model metrics (from TEST set, not validation)
    """
    # Load trained model
    model = create_model(model_name=None, pretrained=False, model_path=model_path)
    
    # Get validation metrics for reference (from validation set)
    val_precision = val_recall = val_mAP50 = val_mAP50_95 = 0
    try:
        val_metrics = model.val(data=dataset_yaml, verbose=False)
        if hasattr(val_metrics, 'box'):
            val_precision = float(getattr(val_metrics.box, 'p', 0))
            val_recall = float(getattr(val_metrics.box, 'r', 0))
            val_mAP50 = float(getattr(val_metrics.box, 'mAP50', 0))
            val_mAP50_95 = float(getattr(val_metrics.box, 'mAP50-95', 0))
    except Exception as e:
        print(f"  ⚠ Validation metrics extraction failed: {e}")
    
    # Evaluate on TEST set with ground truth labels
    # Ultralytics supports test split if defined in YAML
    test_precision = test_recall = test_mAP50 = test_mAP50_95 = 0
    
    # Read original YAML to check if test split exists
    with open(dataset_yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # If test split exists in YAML, try to evaluate on it
    test_yaml_path = None
    if 'test' in yaml_data:
        try:
            # Create a temporary YAML with test as validation split for evaluation
            data_root = Path(dataset_yaml).parent
            test_yaml_path = data_root / "test_data.yaml"
            
            # Create YAML with test images as validation (so model.val() can use it)
            test_yaml_content = {
                'val': yaml_data['test'],  # Use test path as val for evaluation
                'nc': yaml_data.get('nc', 1),
                'names': yaml_data.get('names', ['smoke'])
            }
            
            with open(test_yaml_path, 'w') as f:
                yaml.dump(test_yaml_content, f, default_flow_style=False)
            
            # Evaluate on test set (treated as validation split)
            test_metrics = model.val(data=str(test_yaml_path), verbose=False)
            if hasattr(test_metrics, 'box'):
                test_precision = float(getattr(test_metrics.box, 'p', 0))
                test_recall = float(getattr(test_metrics.box, 'r', 0))
                test_mAP50 = float(getattr(test_metrics.box, 'mAP50', 0))
                test_mAP50_95 = float(getattr(test_metrics.box, 'mAP50-95', 0))
        except Exception as e:
            print(f"  ⚠ Test evaluation failed: {e}")
            print(f"     Using validation metrics as fallback.")
        finally:
            # Clean up temporary YAML
            if test_yaml_path and test_yaml_path.exists():
                test_yaml_path.unlink()
    
    # Count detections on test images (for reference)
    images_with_detections = 0
    total_detections = 0
    
    for img_path in test_images:
        results = model(str(img_path), verbose=False)
        result = results[0]
        num_detections = len(result.boxes)
        
        if num_detections > 0:
            images_with_detections += 1
        total_detections += num_detections
    
    return {
        # Test set metrics (primary)
        'precision': test_precision,
        'recall': test_recall,
        'mAP50': test_mAP50,
        'mAP50_95': test_mAP50_95,
        # Validation set metrics (for reference)
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_mAP50': val_mAP50,
        'val_mAP50_95': val_mAP50_95,
        # Detection counts
        'images_with_detections': images_with_detections,
        'total_detections': total_detections,
        'total_images': len(test_images),
        'model_path': model_path
    }


