# Video Scoring System - Detection Module

## Overview
The **Video Scoring System** is an intelligent middleware solution designed to sit between loss prevention systems (like Purchek) and video review teams. Its primary goal is to assign a **Push-Out Probability Score (POPS)** to video clips, filtering out noise and highlighting high-probability theft events.

**This repository currently focuses on the core detection module**, utilizing a specialized YOLOv26 model to accurately identify key objects (carts, people) in retail environments.

## Model Architecture: YOLOv26
This project utilizes a specialized **YOLOv26** model for object detection.

### Why YOLOv26 vs YOLOv8?
While standard YOLOv8 models are excellent general-purpose detectors, this project requires specific optimizations for the retail environment:
1.  **Environmental Robustness**: The v26 variant is fine-tuned to handle diverse store flooring (shiny tile vs. carpet), fluctuating lighting (glare/dimness), and high-angle fisheye views common in CCTV setups.
2.  **Specialized Classes**: Differentiates between `person` and `shopping cart` with high precision, whereas standard COCO-trained YOLOv8 models often misclassify carts as generic objects or fail to detect them reliably in complex retail scenes.
3.  **Efficiency**: Optimized for edge deployment to reduce bandwidth costs and latency.

## Project Structure
- `code/`: Contains the source code for detection and inference.
    - `inference.py`: Script to run object detection on images.
    - `vision.py`: Vision system logic.
    - `train_detector.py`: Training script for the detector.
- `runs/`: Contains training runs and model checkpoints.
    - `runs/detect/yolov26s_person_cart/weights/best.pt`: **The primary model weights.**
- `dataset/`: (Excluded from git) Contains raw training data.

## Usage

### Prerequisites
- Python 3.8+
- `ultralytics`
- `opencv-python`

### Running Inference
To verify the detection model performance:

```bash
python code/inference.py
```

This script will:
1.  Load the `yolov26s` weights from the `runs/` directory.
2.  Process images from `dataset/yolo_dataset/images/val`.
3.  Generate annotated images with bounding boxes.
4.  Save results to `runs/detect/inference_output/`.
