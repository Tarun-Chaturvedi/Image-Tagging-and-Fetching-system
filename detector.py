import os
from ultralytics import YOLO

# 1. Load the base model
model_name = "yolo11s"
model = YOLO(f"{model_name}.pt")


ov_path = f"{model_name}_openvino_model"
if not os.path.exists(ov_path):
    print("First-time setup: Optimizing model for Intel hardware...")
    model.export(format="openvino", half=True) # half=True is key for my laptop t480

# 3. Load the optimized model

op_model = YOLO(ov_path)

def detect_objects(image_path):
    """Runs AI detection and returns unique labels and confidence."""
    # Use device="GPU" if you have it ,I don't have it
    results = op_model.predict(source=image_path, conf=0.6, save=False, device="cpu")

    detected_tags = []
    for result in results:
        for box in result.boxes:
            
            class_id = int(box.cls[0].item())
            label = op_model.names[class_id]
            confidence = float(box.conf[0].item())
            
            detected_tags.append({
                "label": label, 
                "confidence": round(confidence, 2) # Rounded for readability
            })

    return detected_tags
