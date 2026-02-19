from ultralytics import YOLO
import cv2
import os
import glob

def main():
    # Paths
    # Using the specific weights requested by the user
    weights_path = r'runs/detect/yolov26s_person_cart/weights/best.pt'
    images_dir = r'dataset/yolo_dataset/images/val' 
    output_dir = r'runs/detect/inference_output'
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    # Initialize Model
    print(f"Loading model: {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Get Images
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png'))
    #image_paths = image_paths[:10] # Limit to 10 for demo speed
    
    print(f"Processing {len(image_paths)} images...")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Inference
        results = model(img_path)
        
        for r in results:
            # Plot detections
            im_array = r.plot()
            
            # Save
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, im_array)
            print(f"Saved: {save_path}")

    print(f"\nInference complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
