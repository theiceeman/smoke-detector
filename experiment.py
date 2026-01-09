"""
Smoke Detection Experiment
Compares YOLOv8, YOLOv11, and RT-DETR models
"""

import torch
import numpy as np
from pathlib import Path
import json
import argparse

from models.unified import create_model
from utils.data_loader import load_test_images, prepare_yolo_dataset
from train import train_model
from evaluate import evaluate_baseline, evaluate_trained


def run_smoke_detection_experiment(
    data_root: str = "dataset",
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    device: torch.device = None,
    patience: int = 5,
):
    """
    Run smoke detection experiment comparing YOLOv8, YOLOv11, and RT-DETR.

    Args:
        data_root: Root directory containing dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate (not used for Ultralytics models, they have defaults)
        device: Device to run on (auto-detected if None)
        patience: Early stopping patience
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("SETTING UP SMOKE DETECTION EXPERIMENT")
    print("=" * 80)

    data_root = Path(data_root)
    dataset_yaml = prepare_yolo_dataset(str(data_root))
    test_dir = data_root / "test" / "images"
    
    # Load test images (used for both baseline and evaluation)
    test_images = load_test_images(str(test_dir))
    print(f"  Test set: {len(test_images)} images")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Models to test
    models_to_test = ['yolov8', 'yolov11', 'rtdetr']
    
    # Initialize results storage
    results = {}
    for model_name in models_to_test:
        results[model_name] = {
            "baseline": {},
            "trained": {}
        }
    
    # ========== BASELINE TESTING ==========
    print("\n" + "=" * 80)
    print("BASELINE TESTING (Pre-trained Models)")
    print("=" * 80)

    for model_name in models_to_test:
        print(f"\nTesting {model_name.upper()} (pre-trained)...")
        try:
            model = create_model(model_name, pretrained=True)
            baseline_results = evaluate_baseline(model, test_images)
            results[model_name]["baseline"] = baseline_results
            
            print(f"  Images with detections: {baseline_results['images_with_detections']}/{len(test_images)}")
            print(f"  Detection rate: {baseline_results['detection_rate']:.2%}")
            print(f"  Total detections: {baseline_results['total_detections']}")
        except Exception as e:
            print(f"  ❌ Error testing {model_name} baseline: {e}")
            results[model_name]["baseline"] = {
                "error": str(e),
                "images_with_detections": 0,
                "total_images": len(test_images),
                "detection_rate": 0
            }
    
    # ========== TRAINING ==========
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    print(f"  Epochs: {num_epochs}, Patience: {patience}")
    
    for model_name in models_to_test:
        print(f"\nTraining {model_name.upper()}...")
        try:
            model_path = train_model(
                model_name=model_name,
                dataset_yaml=dataset_yaml,
                num_epochs=num_epochs,
                patience=patience,
                save_dir="checkpoints"
            )
            results[model_name]["model_path"] = model_path
            print(f"  ✓ Training complete. Best model: {model_path}")
        except Exception as e:
            print(f"  ❌ Error training {model_name}: {e}")
            results[model_name]["model_path"] = None
            results[model_name]["training_error"] = str(e)
    
    # ========== EVALUATION ==========
    print("\n" + "=" * 80)
    print("EVALUATING TRAINED MODELS")
    print("=" * 80)
    
    for model_name in models_to_test:
        if results[model_name].get("model_path") is None:
            print(f"\nSkipping {model_name.upper()} (training failed)")
            continue
        
        print(f"\nEvaluating {model_name.upper()} (trained)...")
        try:
            trained_results = evaluate_trained(
                model_path=results[model_name]["model_path"],
                dataset_yaml=dataset_yaml,
                test_images=test_images
            )
            results[model_name]["trained"] = trained_results
            
            print(f"  TEST SET METRICS (14 images):")
            print(f"    Precision: {trained_results['precision']:.2%}")
            print(f"    Recall: {trained_results['recall']:.2%}")
            print(f"    mAP@0.5: {trained_results['mAP50']:.2%}")
            print(f"    mAP@0.5:0.95: {trained_results['mAP50_95']:.2%}")
            print(f"    Images with detections: {trained_results['images_with_detections']}/{len(test_images)}")
            
            if trained_results.get('val_precision') is not None:
                print(f"  VALIDATION SET METRICS (10 images, for reference):")
                print(f"    Precision: {trained_results['val_precision']:.2%}")
                print(f"    Recall: {trained_results['val_recall']:.2%}")
                print(f"    mAP@0.5: {trained_results['val_mAP50']:.2%}")
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            results[model_name]["trained"] = {"error": str(e)}
    
    # ========== COMPARISON ==========
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for model_name in models_to_test:
        baseline = results[model_name].get("baseline", {})
        trained = results[model_name].get("trained", {})
        
        if "error" in baseline or "error" in trained:
            print(f"\n{model_name.upper()}: Skipped (error occurred)")
            continue
        
        baseline_detections = baseline.get("images_with_detections", 0)
        trained_detections = trained.get("images_with_detections", 0)
        improvement = trained_detections - baseline_detections
        
        print(f"\n{model_name.upper()}:")
        print(f"  Baseline detections: {baseline_detections}/{len(test_images)}")
        print(f"  Trained detections: {trained_detections}/{len(test_images)}")
        print(f"  Improvement: +{improvement} images")
        
        if trained.get("precision") is not None:
            print(f"  Test Precision: {trained['precision']:.2%}")
            print(f"  Test Recall: {trained['recall']:.2%}")
            print(f"  Test mAP@0.5: {trained['mAP50']:.2%}")
            if trained.get('val_precision') is not None:
                print(f"  Val Precision: {trained['val_precision']:.2%} (for reference)")

    # ========== SAVE RESULTS ==========
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Prepare results for JSON (convert Path objects and handle errors)
    results_json = {}
    for model_name in models_to_test:
        results_json[model_name] = {}
        
        # Baseline results
        baseline = results[model_name].get("baseline", {})
        if "error" not in baseline:
            results_json[model_name]["baseline"] = {
                "images_with_detections": int(baseline.get("images_with_detections", 0)),
                "total_images": int(baseline.get("total_images", len(test_images))),
                "detection_rate": float(baseline.get("detection_rate", 0)),
                "total_detections": int(baseline.get("total_detections", 0))
            }
        else:
            results_json[model_name]["baseline"] = {"error": baseline["error"]}
        
        # Trained results
        trained = results[model_name].get("trained", {})
        if "error" not in trained and trained:
            results_json[model_name]["trained"] = {
                "precision": float(trained.get("precision", 0)),
                "recall": float(trained.get("recall", 0)),
                "mAP50": float(trained.get("mAP50", 0)),
                "mAP50_95": float(trained.get("mAP50_95", 0)),
                "images_with_detections": int(trained.get("images_with_detections", 0)),
                "total_images": int(trained.get("total_images", len(test_images))),
                "improvement": int(trained.get("images_with_detections", 0) - baseline.get("images_with_detections", 0)),
                "model_path": str(trained.get("model_path", ""))
            }
        else:
            results_json[model_name]["trained"] = {"error": trained.get("error", "Unknown error")}
    
    # Save to JSON
    with open(results_dir / "detailed_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\nResults saved to:")
    print(f"  - {results_dir / 'detailed_results.json'}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke detection experiment")
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Root directory containing dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (not used for Ultralytics)")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    args = parser.parse_args()

    run_smoke_detection_experiment(
        data_root=args.data_root,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
    )
