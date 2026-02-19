import os
import glob
import shutil
import random

def manual_split():
    base_dir = "d:/gatekeeper_projects/empty_or_full_classification/dataset/yolo_dataset"
    train_img_dir = os.path.join(base_dir, "images/train")
    val_img_dir = os.path.join(base_dir, "images/val")
    train_lbl_dir = os.path.join(base_dir, "labels/train")
    val_lbl_dir = os.path.join(base_dir, "labels/val")

    # Ensure val dirs exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Get all images
    images = glob.glob(os.path.join(train_img_dir, "*.jpg"))
    if not images:
        print("No images found in train dir.")
        return

    print(f"Found {len(images)} images in train.")
    
    # Check if val is already populated
    val_images = glob.glob(os.path.join(val_img_dir, "*.jpg"))
    if len(val_images) > 100:
        print(f"Validation set already has {len(val_images)} images. Skipping split.")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(images)
    
    # Move 10% to val (1800 images is enough)
    num_val = int(len(images) * 0.1)
    to_move = images[:num_val]
    
    print(f"Moving {len(to_move)} images to validation set...")
    
    count = 0
    for img_path in to_move:
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        
        # Move Image
        shutil.move(img_path, os.path.join(val_img_dir, basename))
        
        # Move Label
        lbl_path = os.path.join(train_lbl_dir, f"{stem}.txt")
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(val_lbl_dir, f"{stem}.txt"))
        
        count += 1
        
    print(f"Moved {count} pairs.")

if __name__ == "__main__":
    manual_split()
