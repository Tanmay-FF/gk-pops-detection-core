from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def train_detector():

    try:
        model = YOLO(r'D:\gatekeeper_projects\empty_or_full_classification\code\yolo26s.pt')
    except Exception as e:
        print(f"Could not load yolov26s.pt directly: {e}")
        print("Attempting to load standard yolov8s.pt as a placeholder/fallback or creating new model.")
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
