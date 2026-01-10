# python3 09-project/yolo.py
""" 
YOLOv8 detection on a single image. 
we used it for the first test we did on yolo to know how well it performs with smoke. 
"""
from ultralytics import YOLO
import cv2
import os
import json

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load the pretrained YOLO model
model = YOLO("yolov8s.pt")   # small model; good for CPU too

# 2. Load one of your frames
img_path = os.path.join(script_dir, "extracted_frames", "test.png")
img = cv2.imread(img_path)

# Check if image loaded successfully
if img is None:
    print(f"Error: Could not load image from {img_path}")
    print("Please check that the file exists.")
    exit(1)

print(f"Analyzing image: {img_path}")
print(f"Image size: {img.shape[1]}x{img.shape[0]}")

# 3. Run YOLO on the image
results = model(img)

# 4. Print what YOLO detected and store results
print("\n=== YOLO Detection Results ===")
detections = []
result = results[0]
boxes = result.boxes

if len(boxes) > 0:
    print(f"Found {len(boxes)} object(s):")
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names[cls]
        
        # Get bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Store detection data
        detection = {
            "class_id": int(cls),
            "class_name": class_name,
            "confidence": float(conf),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            }
        }
        detections.append(detection)
        
        print(f"  {i+1}. {class_name}: {conf:.2%} confidence")
else:
    print("No objects detected.")

# 5. Save detection results to JSON file
output_dir = os.path.join(script_dir, "yolo_results")
os.makedirs(output_dir, exist_ok=True)

# Get base filename without extension
base_name = os.path.splitext(os.path.basename(img_path))[0]
json_path = os.path.join(output_dir, f"{base_name}_detections.json")

results_data = {
    "image_path": img_path,
    "image_size": {
        "width": int(img.shape[1]),
        "height": int(img.shape[0])
    },
    "num_detections": len(detections),
    "detections": detections
}

with open(json_path, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\nDetection data saved to: {json_path}")

# 6. Draw detections on the image
annotated = results[0].plot()

# 7. Save annotated image
annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
cv2.imwrite(annotated_path, annotated)
print(f"Annotated image saved to: {annotated_path}")

# 8. Show the output
cv2.imshow("YOLO Detection", annotated)
print("\nPress any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
