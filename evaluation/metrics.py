# evaluation/metrics.py
import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA (list or np.array): The first bounding box, formatted as [x1, y1, x2, y2].
        boxB (list or np.array): The second bounding box, formatted as [x1, y1, x2, y2].

    Returns:
        float: The IoU score, a value between 0 and 1.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    # If the boxes do not overlap, the intersection area will be negative.
    # In such cases, the intersection area should be 0.
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU by taking the intersection area and dividing it by the sum
    # of the two areas minus the intersection area (to avoid double-counting).
    union_area = float(boxA_area + boxB_area - intersection_area)

    # Handle the case where the union area is zero
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou

# This block allows us to test the function directly
if __name__ == '__main__':
    # Example boxes
    box_ground_truth = [100, 100, 200, 200]
    box_prediction_good = [110, 110, 210, 210] # Good overlap
    box_prediction_bad = [300, 300, 400, 400] # No overlap

    iou_good = calculate_iou(box_ground_truth, box_prediction_good)
    iou_bad = calculate_iou(box_ground_truth, box_prediction_bad)

    print(f"Ground Truth Box: {box_ground_truth}")
    print(f"Good Prediction Box: {box_prediction_good}")
    print(f"Bad Prediction Box: {box_prediction_bad}")
    print("-" * 30)
    print(f"IoU for good prediction: {iou_good:.4f}") # Should be high (e.g., > 0.7)
    print(f"IoU for bad prediction: {iou_bad:.4f}")   # Should be 0.0