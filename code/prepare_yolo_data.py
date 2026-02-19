import json
import cv2
import os
import glob
import random
from pathlib import Path
import shutil

# Classes
CLASS_MAPPING = {
    'person': 0,
    'cart': 1
}

def setup_yolo_dirs(base_dir):
    """Create YOLO directory structure."""
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def convert_to_yolo_bbox(bbox, img_w, img_h):
    """Convert x1, y1, x2, y2 to normalized x_center, y_center, width, height."""
    x1 = bbox['x1']
    y1 = bbox['y1']
    x2 = bbox['x2']
    y2 = bbox['y2']
    
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2.0
    y_center = y1 + h / 2.0
    
    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh
    
    return x, y, w, h

def process_video_and_json(json_path, video_path, output_root, split='train'):
    """
    Process a single video and its json annotation.
    Extract annotated frames and save labels.
    """
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    video_info = data.get('video_info', {})
    img_width = video_info.get('width', 1280)
    img_height = video_info.get('height', 720)
    video_name = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    frame_dict = data.get('frames', {})
    annotated_frame_numbers = sorted([int(k) for k in frame_dict.keys()])

    images_dir = os.path.join(output_root, 'images', split)
    labels_dir = os.path.join(output_root, 'labels', split)

    count = 0
    for frame_num in annotated_frame_numbers:
        # JSON frames are 1-based, video 0-based
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        
        if not ret:
            continue

        frame_data = frame_dict[str(frame_num)]
        
        # Collect YOLO labels for this frame
        yolo_labels = []
        
        # Process People
        people = frame_data.get('people', {})
        for pid, p_info in people.items():
            bbox = p_info.get('bbox')
            if bbox:
                x, y, w, h = convert_to_yolo_bbox(bbox, img_width, img_height)
                yolo_labels.append(f"{CLASS_MAPPING['person']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # Process Carts
        carts = frame_data.get('carts', {})
        for cid, c_info in carts.items():
            bbox = c_info.get('bbox')
            if bbox:
                x, y, w, h = convert_to_yolo_bbox(bbox, img_width, img_height)
                yolo_labels.append(f"{CLASS_MAPPING['cart']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        if not yolo_labels:
            continue

        # Save Image
        img_filename = f"{video_name}_frame_{frame_num}.jpg"
        cv2.imwrite(os.path.join(images_dir, img_filename), frame)
        
        # Save Label
        label_filename = f"{video_name}_frame_{frame_num}.txt"
        with open(os.path.join(labels_dir, label_filename), 'w') as lf:
            lf.write("\n".join(yolo_labels))
            
        count += 1

    cap.release()
    return count

def create_data_yaml(output_root):
    content = f"""
path: {output_root}
train: images/train
val: images/val
names:
  0: person
  1: cart
"""
    with open(os.path.join(output_root, 'data.yaml'), 'w') as f:
        f.write(content.strip())

def main():
    dataset_root = "d:/gatekeeper_projects/empty_or_full_classification/dataset"
    yolo_root = os.path.join(dataset_root, "yolo_dataset")
    
    if os.path.exists(yolo_root):
        try:
            shutil.rmtree(yolo_root, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not fully clean {yolo_root}: {e}")
            
    setup_yolo_dirs(yolo_root)
    
    # Gather Data
    # We will use both 'empty' and 'full' folders as source
    # And split by video file
    
    all_pairs = [] # (json_path, video_path)
    
    for subdir in ['empty', 'full']:
        json_dir = os.path.join(dataset_root, subdir, 'json')
        video_dir = os.path.join(dataset_root, subdir, 'pcvideo')
        
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        for jf in json_files:
            basename = Path(jf).stem
            vf = os.path.join(video_dir, f"{basename}.mp4")
            if os.path.exists(vf):
                all_pairs.append((jf, vf))
    
    # Shuffle and Split (80/20)
    random.seed(42)
    random.shuffle(all_pairs)
    
    split_idx = int(0.8 * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    print(f"Found {len(all_pairs)} video pairs. Training on {len(train_pairs)}, Validating on {len(val_pairs)}")
    
    # Process Train
    print("Processing Training Set...")
    total_train_frames = 0
    for jf, vf in train_pairs:
        total_train_frames += process_video_and_json(jf, vf, yolo_root, 'train')
    
    # Process Val
    print("Processing Validation Set...")
    total_val_frames = 0
    for jf, vf in val_pairs:
        total_val_frames += process_video_and_json(jf, vf, yolo_root, 'val')
        
    print(f"Done. Extracted {total_train_frames} training frames and {total_val_frames} validation frames.")
    
    # Create config
    create_data_yaml(yolo_root)
    print(f"Created data.yaml at {os.path.join(yolo_root, 'data.yaml')}")

if __name__ == "__main__":
    main()
