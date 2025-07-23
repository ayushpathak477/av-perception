# evaluation/debug_boxes.py
import os
import cv2
import torch
from ultralytics import YOLO
from .metrics import calculate_iou # CORRECTED: Added a dot for relative import

def load_kitti_labels(label_path):
    """Loads ground truth labels from a KITTI label file."""
    ground_truths = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            # Standardize to lowercase
            obj_class = parts[0].lower()
            bbox = [float(p) for p in parts[4:8]]
            ground_truths.append({"class": obj_class, "bbox": bbox})
    return ground_truths

if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_PATH = 'data/2011_09_26_drive_0013_sync/image_02/data/0000000000.png'
    LABEL_PATH = 'data/training_sample/label_2/0000000000.txt'
    
    # Load the model
    model = YOLO('yolov8n.pt')

    # Load the image and labels
    image = cv2.imread(IMAGE_PATH)
    ground_truths = load_kitti_labels(LABEL_PATH)
    
    # Run detector
    preds = model(IMAGE_PATH, verbose=False)[0]

    # --- Print the exact prediction coordinates ---
    print("\n--- YOLO's Predicted Bounding Boxes ---")
    for box in preds.boxes:
        class_name = preds.names[int(box.cls[0])]
        if class_name == 'car': # Only show cars
            coords = box.xyxy[0].cpu().numpy()
            print(f"Car: {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f} {coords[3]:.2f}")
    print("----------------------------------------\n")

    # --- Visualization ---
    # Draw Ground Truth boxes in RED
    for gt in ground_truths:
        x1, y1, x2, y2 = map(int, gt['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"GT: {gt['class']}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw Predicted boxes in GREEN
    for box in preds.boxes:
        pred_bbox = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, pred_bbox)
        class_name = preds.names[int(box.cls[0])]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Pred: {class_name}", (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result
    print("Displaying debug image. Press any key to close.")
    cv2.imshow("Ground Truth (Red) vs. Predictions (Green)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()