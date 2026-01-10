"""
Smoke Detection Experiment
Compares YOLOv8, YOLOv11, and RT-DETR models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from pathlib import Path
import json
import argparse

from models.unified import create_model
from utils.data_loader import load_test_images, prepare_yolo_dataset
from v2.train import train_model
from evaluate import evaluate_baseline, evaluate_trained


def run_smoke_detection_experiment(
    data_root: str = "dataset",
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    device: torch.device = None,
    patience: int = 50,
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

    # setup experiment
    data_root = Path(data_root)
    dataset_yaml = prepare_yolo_dataset(str(data_root))

    test_dir = data_root / "test" / "images"
    test_images = load_test_images(str(test_dir))

    print(f"  Test set: {len(test_images)} images")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    models_to_test = [
        "yolov8",
        #    'yolov11', 'rtdetr'
    ]

    # Initialize results storage
    results = {}
    for model_name in models_to_test:
        results[model_name] = {"baseline": {}, "trained": {}}

    # BASELINE TESTING
    print(">> BASELINE TESTING (Pre-trained Models) <<")

    for model_name in models_to_test:
        print(f"\nTesting {model_name.upper()} (pre-trained)...")
        try:
            model = create_model(model_name, pretrained=True)
            baseline_results = evaluate_baseline(model, test_images)
            results[model_name]["baseline"] = baseline_results
        except Exception as e:
            print(f"  ❌ Error testing {model_name} baseline: {e}")

    print("Results from baseline testing:", results)


    # TRAINING
    print(">> TRAINING MODELS <<")

    for model_name in models_to_test:
        print(f"\nTraining {model_name.upper()}...")
        try:
            model_path = train_model(
                model_name=model_name,
                dataset_yaml=dataset_yaml,
                num_epochs=num_epochs,
                patience=patience,
                save_dir="checkpoints",
            )
            results[model_name]["model_path"] = model_path
            print(f"  ✓ Training complete. Best model: {model_path}")
        except Exception as e:
            print(f"  ❌ Error training {model_name}: {e}")
            results[model_name]["model_path"] = None
            results[model_name]["training_error"] = str(e)


    # EVALUATION
    print(">> EVALUATING TRAINED MODELS <<")

    for model_name in models_to_test:
        if results[model_name].get("model_path") is None:
            print(f"\nSkipping {model_name.upper()} (training failed)")
            results[model_name]["trained"] = {"error": results[model_name].get("training_error", "Training failed")}
            continue

        print(f"\nEvaluating {model_name.upper()} (trained)...")
        try:
            trained_results = evaluate_trained(
                model_path=results[model_name]["model_path"],
                dataset_yaml=dataset_yaml,
                test_images=test_images,
            )
            results[model_name]["trained"] = trained_results
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            results[model_name]["trained"] = {"error": str(e)}


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
                "images_with_detections": int(
                    baseline.get("images_with_detections", 0)
                ),
                "total_images": int(baseline.get("total_images", len(test_images))),
                "detection_rate": float(baseline.get("detection_rate", 0)),
                "total_detections": int(baseline.get("total_detections", 0)),
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
                "improvement": int(
                    trained.get("images_with_detections", 0)
                    - baseline.get("images_with_detections", 0)
                ),
                "model_path": str(trained.get("model_path", "")),
            }
        else:
            error_msg = trained.get("error", "Unknown error")
            results_json[model_name]["trained"] = {"error": error_msg}
            if error_msg != "Unknown error":
                print(f"  ⚠ {model_name.upper()}: {error_msg}")

    # Save to JSON
    with open(results_dir / "detailed_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to:  - {results_dir / 'detailed_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke detection experiment")
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Root directory containing dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience"
    )

    args = parser.parse_args()

    run_smoke_detection_experiment(
        data_root=args.data_root,
        num_epochs=args.epochs,
        patience=args.patience,
    )
