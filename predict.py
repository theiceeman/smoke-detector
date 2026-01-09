# python3 09-project/predict.py
from ultralytics import YOLO
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load your custom trained model
model = YOLO(os.path.join(script_dir, "runs/detect/train/weights/best.pt"))

# Run predictions on test frames
test_frames_path = os.path.join(script_dir, "dataset/test_frames")

# Configure save location: project parameter sets the base directory, name sets the subfolder
# This ensures results are saved in the project directory regardless of where script is run from
results = model.predict(
    source=test_frames_path, 
    save=True, 
    save_txt=True,
    project=script_dir,  # Base directory for saving results
    name="predictions"    # Creates: {script_dir}/predictions/
)

# Get the actual save directory from results
save_dir = results[0].save_dir if results else os.path.join(script_dir, "predictions")
print(f"Predictions completed! Results saved to: {save_dir}")
