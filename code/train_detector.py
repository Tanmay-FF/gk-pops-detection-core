from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def train_detector():
    # Initialize YOLOv26s model
    # Note: If yolov26s.pt is not available locally or on hub, this might raise an error 
    # unless it's a custom model file provided by the user. 
    # Assuming 'yolov26s.pt' is the requested model name.
    #model = YOLO('yolov26s.yaml')  # Load model structure (if v26 is custom) or 'yolov26s.pt'
    
    # Or if standard YOLOv8 is acceptable as fallback if v26 assumes v8 architecture:
    # model = YOLO('yolov8s.pt') 
    
    # Given the explicit user request for v26s, we try to load it. 
    # If it's a non-standard model not in ultralytics, we might need a yaml definition.
    # For now, using the string identifier.
    try:
        model = YOLO(r'D:\gatekeeper_projects\empty_or_full_classification\code\yolo26s.pt')
    except Exception as e:
        print(f"Could not load yolov26s.pt directly: {e}")
        print("Attempting to load standard yolov8s.pt as a placeholder/fallback or creating new model.")
        # Fallback to creating a new model from scratch or v8 if v26 fails, 
        # but let's try to verify if we should just use a standard one.
        # For this script, I'll stick to the requested name.
        model = YOLO('yolov26s.pt')

    # Train the model
    # Use absolute path for data.yaml to avoid ambiguity
    data_config = os.path.abspath('dataset/yolo_dataset/data.yaml')
    
    results = model.train(
        data=data_config,
        epochs=20,
        imgsz=640,
        batch=32,
        name='yolov26s_person_cart',
        device=0,
        patience=20,
        save=True
    )

if __name__ == '__main__':
    train_detector()
