# GK POPS Detection Core

## Overview

**GK POPS Detection Core** is the detection and scoring engine for the Gatekeeper Video Scoring System — an intelligent middleware solution designed to sit between retail loss prevention systems and video review teams.

Its primary goal is to assign a **Push-Out Probability Score (POPS)** to video clips from retail CCTV footage, filtering out noise and surfacing high-probability theft/push-out events for human review.

This repository covers the **core detection and tracking module**: detecting persons and shopping carts, tracking them across frames, linking persons to carts, and analysing their motion. POPS scoring and event classification are built on top of the outputs produced here.

---

## System Architecture

```
Raw CCTV Video
      |
      v
[YOLOv26 Detection] --> persons + carts with bounding boxes
      |
      v
[BoTSORT Tracker] --> persistent display IDs across frames
      |
      v
[Motion Analyser] --> speed, direction, acceleration, INBOUND/OUTBOUND label
      |
      v
[Person-Cart Linker] --> proximity + IoU-based associations (2.75 s confirmation)
      |
      v
[JSON Output] --> per-frame tracking data (positions, motion, links)
      +
[Annotated MP4] --> bounding boxes, trails, link lines, motion overlays
```

---

## Model Architecture: YOLOv26

This project uses a specialized **YOLOv26** model for object detection with two trained variants:

| Variant | Weights File | Use Case |
|---|---|---|
| YOLOv26s (small) | `runs/detect/yolov26s_person_cart/weights/best.pt` | Edge deployment, lower latency |
| YOLOv26m (medium) | `runs/detect/yolov26m_person_cart/weights/best.pt` | Higher accuracy |

**Detected Classes:**
- `0` — person
- `1` — cart

### Why YOLOv26 over YOLOv8?

1. **Environmental Robustness** — Fine-tuned for diverse retail conditions: shiny tile vs carpet, variable lighting (glare/dimness), and high-angle fisheye CCTV views.
2. **Specialized Classes** — Optimized specifically for `person` and `shopping cart` detection. Standard COCO-trained YOLOv8 often misclassifies carts in complex retail scenes.
3. **Edge Efficiency** — Optimized for edge deployment to reduce bandwidth costs and latency.

---

## Project Structure

```
gk-pops-detection-core/
├── code/
│   ├── train_detector.py        # Train YOLOv26 on prepared dataset
│   ├── inference.py             # Run inference on validation images
│   ├── prepare_yolo_data.py     # Convert JSON annotations + videos to YOLO format
│   ├── split_manual.py          # Split training data into train/val sets
│   ├── auto_annotate.py         # Auto-generate pseudo-labels from unlabeled videos
│   ├── object-tracking-plain.py # Multi-object tracking with POPS scoring pipeline
│   ├── demo_app.py              # Interactive Gradio web app for POPS demo
│   └── botsort_retail.yaml      # BoTSORT tracker config tuned for retail CCTV
├── resources/
│   └── sample_annotation.json   # Example annotation format
├── runs/
│   └── detect/
│       ├── yolov26s_person_cart/ # Small model checkpoints + training results
│       └── yolov26m_person_cart/ # Medium model checkpoints + training results
└── dataset/                     # (Excluded from git) Raw training data
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (recommended for GPU acceleration)
- FFmpeg (required by `demo_app.py`)

### Install Dependencies

```bash
pip install ultralytics opencv-python torch torchvision numpy gradio imageio-ffmpeg pillow tqdm
```

---

## Usage

### 1. Data Preparation

Place raw retail CCTV videos and their JSON annotation files in the following structure:

```
dataset/
├── empty/
│   ├── json/        # JSON annotation files
│   └── pcvideo/     # Video files
└── full/
    ├── json/
    └── pcvideo/
```

Then convert to YOLO format:

```bash
python code/prepare_yolo_data.py
```

This extracts annotated frames from videos, converts bounding boxes to YOLO normalized format, and generates an 80/20 train/val split with a `data.yaml` config file.

To manually adjust the split afterward:

```bash
python code/split_manual.py
```

### 2. Training

```bash
python code/train_detector.py
```

Training settings (configurable in script):
- **Model:** YOLOv26m (falls back to YOLOv8m if unavailable)
- **Epochs:** 30
- **Batch size:** 16
- **Image size:** 640x640
- **Early stopping patience:** 20 epochs
- **Device:** CUDA GPU 0

Trained weights are saved to `runs/detect/`.

### 3. Inference

Run the trained model on validation images:

```bash
python code/inference.py
```

Loads `yolov26s_person_cart` weights, processes images from `dataset/yolo_dataset/images/val`, and saves annotated results to `runs/detect/inference_output/`.

### 4. Auto-Annotation (Semi-Supervised Expansion)

Generate pseudo-labels for unlabeled videos to expand the training dataset:

```bash
python code/auto_annotate.py \
  --input_dir /path/to/unlabeled/videos \
  --model runs/detect/yolov26s_person_cart/weights/best.pt \
  --output auto_dataset \
  --interval 2.0
```

| Argument | Description | Default |
|---|---|---|
| `--input_dir` | Folder containing video files (.mp4, .avi, .mkv, .mov) | Required |
| `--model` | Path to trained YOLO weights | `yolov26m_person_cart` |
| `--output` | Output directory for images and YOLO labels | Required |
| `--interval` | Frame extraction interval in seconds | `1.0` |

Only frames with detections are saved. Output is ready for retraining.

### 5. Object Tracking Pipeline

Run the standalone tracking pipeline on a video:

```bash
python code/object-tracking-plain.py
```

This performs:
- YOLOv26 detection + BoTSORT tracking per frame
- Independent sequential display IDs for persons and carts
- Proximity-based person-cart linking
- Motion metrics (speed, direction, acceleration)
- JSON output with per-frame tracking details

### 6. Interactive Demo App

Launch the Gradio web UI for interactive video analysis:

```bash
python code/demo_app.py
```

The demo app runs the full detection, tracking, and linking pipeline and provides:
- Video upload + camera placement selector (inside/outside facing entrance)
- YOLOv26m detection with BoTSORT tracking
- Person-cart link detection with 2.75 s confirmation window
- Per-object motion overlays: speed status, INBOUND/OUTBOUND direction, link indicator
- Magenta link lines drawn between confirmed person-cart pairs
- Frame counter + live person/cart/link counts overlaid on video
- Annotated MP4 download + full per-frame JSON download
- Info tabs: Video Info, Detection Summary, Model Configuration, Annotation Legend

**Annotation colours:**

| Colour | Meaning |
|---|---|
| Green box | Person |
| Orange box | Cart |
| Magenta line | Confirmed person-cart link |
| Red text | Outbound + Fast movement |
| Orange text | Outbound movement |
| Cyan/yellow text | Inbound movement |

Access the app at `http://localhost:7860` after launch.

---

## BoTSORT Tracker Configuration

The tracker is pre-tuned for retail CCTV in `code/botsort_retail.yaml`:

| Parameter | Value | Reason |
|---|---|---|
| `track_high_thresh` | 0.3 | Lower threshold to catch partially-visible carts |
| `track_low_thresh` | 0.1 | ByteTrack-style fallback for low-confidence detections |
| `track_buffer` | 120 frames | Handles long occlusions (e.g., cart behind shelving) |
| `match_thresh` | 0.7 | IoU matching threshold |
| `gmc_method` | sparseOptFlow | Camera motion compensation |
| Re-ID | Disabled | Too compute-heavy for edge deployment |

---

## JSON Output Format

`demo_app.py` and `object-tracking-plain.py` both produce a structured JSON file with per-frame tracking data. See `resources/sample_annotation.json` for a full example. Key structure:

```json
{
  "video_info": { "video_name": "...", "width": 1280, "height": 720, "fps": 20.0, "total_frames": 400 },
  "frames": {
    "1": {
      "frame_number": 1,
      "timestamp": 0.05,
      "people": {
        "P1": {
          "id": 1,
          "centroid": { "x": 640.0, "y": 360.0 },
          "bbox": { "x1": 600, "y1": 200, "x2": 680, "y2": 520, "width": 80, "height": 320 },
          "motion": { "speed": 45.2, "direction": 270.0, "direction_label": "OUTBOUND", "speed_status": "SLOW", "acceleration": 0.5 },
          "tracking": { "positions_history": [...], "speed_history": [...], "yolo_confidence": 0.91 },
          "linking": { "is_linked": true, "linked_cart_id": 1, "link_confidence": 0.0 }
        }
      },
      "carts": { "C1": { ... } },
      "links": {
        "P1_C1": { "person_id": 1, "cart_id": 1, "distance": 85.3, "established_frame": 12, "duration_frames": 34 }
      },
      "statistics": { "total_people": 1, "total_carts": 1, "active_links": 1, "people_disappeared": 0, "carts_disappeared": 0 }
    }
  },
  "summary": { "total_people_seen": 3, "total_carts_seen": 2, "total_links_established": 2 },
  "processing_info": { "device": "cuda:0", "model": "YOLOv26m", "tracker": "BoTSORT" }
}
```

---

## Dataset

- **Source:** Retail CCTV footage with manual JSON annotations
- **Classes:** `empty` cart videos and `full` cart videos
- **Resolution:** 1280x720 (standard CCTV)
- **Frame rate:** ~20 FPS
- **Dataset directory is excluded from git** (contains proprietary footage)

---

## License

Private — Gatekeeper internal use only.
