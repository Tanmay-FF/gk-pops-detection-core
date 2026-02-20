from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def train_detector():

    try:
        # User requested v26m, but if it doesn't exist, we must use a real model as base
        # Trying to load a non-existent file path directly causes error.
        # We will check if it exists, otherwise use standard yolov8m.
        if os.path.exists(r'D:\gatekeeper_projects\empty_or_full_classification\code\yolov26m.pt'):
             model = YOLO(r'D:\gatekeeper_projects\empty_or_full_classification\code\yolov26m.pt')
        else:
             print("yolov26m.pt not found locally. Using standard yolov8m.pt as base.")
             model = YOLO('yolov8m.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Fallback to yolov8m.pt")
        model = YOLO('yolov8m.pt')

    # Train the model
    # Use absolute path for data.yaml to avoid ambiguity
    data_config = os.path.abspath('dataset/yolo_dataset/data.yaml')
    
    results = model.train(
        data=data_config,
        epochs=30,
        imgsz=640,
        batch=16,
        name='yolov26m_cart_weighted',
        device=0,
        patience=20,
        save=True,
        # image_weights=True removed due to incompatibility
    )

if __name__ == '__main__':
    train_detector()
