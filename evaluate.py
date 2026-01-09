"""Unified evaluation functions for all Ultralytics models"""

from pathlib import Path
from typing import List, Dict
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
    Evaluate trained model on test set.
    
    Args:
        model_path: Path to trained model weights
        dataset_yaml: Path to dataset YAML file
        test_images: List of test image paths
    
    Returns:
        Dictionary with trained model metrics
    """
    # Load trained model
    model = create_model(model_name=None, pretrained=False, model_path=model_path)
    
    # Run validation to get metrics
    try:
        metrics = model.val(data=dataset_yaml, verbose=False)
        
        # Extract metrics
        if hasattr(metrics, 'box'):
            precision = float(getattr(metrics.box, 'p', 0))
            recall = float(getattr(metrics.box, 'r', 0))
            mAP50 = float(getattr(metrics.box, 'mAP50', 0))
            mAP50_95 = float(getattr(metrics.box, 'mAP50-95', 0))
        else:
            # Fallback if metrics structure is different
            precision = 0
            recall = 0
            mAP50 = 0
            mAP50_95 = 0
    except Exception as e:
        print(f"  ⚠ Validation metrics extraction failed: {e}")
        precision = recall = mAP50 = mAP50_95 = 0
    
    # Test on individual images
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
        'precision': precision,
        'recall': recall,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'images_with_detections': images_with_detections,
        'total_detections': total_detections,
        'total_images': len(test_images),
        'model_path': model_path
    }


