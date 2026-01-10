# python3 09-project/predict.py
from ultralytics import YOLO
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(script_dir, "runs/detect/train/weights/best.pt"))
test_frames_path = os.path.join(script_dir, "dataset/test_frames")


results = model.predict(
    source=test_frames_path, 
    save=True, 
    save_txt=True,
    project=script_dir,  # Base directory for saving results
    name="predictions"    # Creates: {script_dir}/predictions/
)

save_dir = results[0].save_dir if results else os.path.join(script_dir, "predictions")
print(f"Predictions completed! Results saved to: {save_dir}")
