# evaluation/evaluate_detector.py
import os
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from .metrics import calculate_iou # Use relative import

def load_kitti_labels(label_path):
    """Loads ground truth labels from a KITTI label file."""
    ground_truths = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            # --- CHANGE: Convert class to lowercase immediately ---
            obj_class = parts[0].lower() 
            bbox = [float(p) for p in parts[4:8]]
            ground_truths.append({"class": obj_class, "bbox": bbox})
    return ground_truths

def calculate_ap(rec, prec):
    """Calculates Average Precision (AP) from recall and precision arrays."""
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

if __name__ == '__main__':
    # --- Configuration for our small sample ---
    IMAGE_DIR = 'data/2011_09_26_drive_0013_sync/image_02/data/'
    LABEL_DIR = 'data/training_sample/label_2/'
    IOU_THRESHOLD = 0.5
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO('yolov8n.pt')
    model.to(device)

    predictions_by_class = {}
    ground_truths_by_class = {}

    print("Running evaluation on our sample data...")
    img_file = '0000000000.png'
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, '0000000000.txt')

    # 1. Get model predictions
    preds = model(img_path, verbose=False)[0]
    
    # 2. Load ground truth labels
    ground_truths = load_kitti_labels(label_path)
    
    # 3. Organize predictions by class
    for box in preds.boxes:
        # --- CHANGE: YOLO class names are already lowercase, so no change needed here ---
        class_name = preds.names[int(box.cls[0])]
        if class_name not in predictions_by_class:
            predictions_by_class[class_name] = []
        predictions_by_class[class_name].append({
            "confidence": box.conf[0],
            "bbox": box.xyxy[0].cpu().numpy()
        })
        
    # 4. Organize ground truths by class
    for gt in ground_truths:
        class_name = gt['class']
        if class_name not in ground_truths_by_class:
            ground_truths_by_class[class_name] = []
        ground_truths_by_class[class_name].append(gt['bbox'])

    # --- Calculate mAP ---
    average_precisions = {}
    
    print("\nCalculating Average Precision for each class...")
    # --- CHANGE: Iterate over a combined set of all classes found ---
    all_classes = set(predictions_by_class.keys()) | set(ground_truths_by_class.keys())

    for class_name in all_classes:
        gts = ground_truths_by_class.get(class_name, [])
        class_preds = sorted(predictions_by_class.get(class_name, []), 
                             key=lambda x: x['confidence'], reverse=True)
        
        if not gts:
            # If there are no ground truths, AP is 0 for this class if there are predictions
            ap = 0.0 if class_preds else 1.0 
            average_precisions[class_name] = ap
            print(f"- AP for '{class_name}': {ap:.4f} (No ground truth)")
            continue

        num_gts = len(gts)
        true_positives = np.zeros(len(class_preds))
        false_positives = np.zeros(len(class_preds))
        detected_gts = [False] * num_gts

        for i, pred in enumerate(class_preds):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gts):
                iou = calculate_iou(pred['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= IOU_THRESHOLD:
                if not detected_gts[best_gt_idx]:
                    true_positives[i] = 1
                    detected_gts[best_gt_idx] = True
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1

        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        recall = tp_cumsum / num_gts if num_gts > 0 else np.zeros_like(tp_cumsum)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum).any() > 0 else np.zeros_like(tp_cumsum)
        
        ap = calculate_ap(recall, precision)
        average_precisions[class_name] = ap
        print(f"- AP for '{class_name}': {ap:.4f}")

    mAP = np.mean(list(average_precisions.values())) if average_precisions else 0.0
    
    print("\n" + "="*30)
    print(f"Final mAP @ {IOU_THRESHOLD} IoU: {mAP:.4f}")
    print("="*30)