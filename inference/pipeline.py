import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from data.calibration import KittiCalibration

class AVPerceptionPipeline:
    def __init__(self, cam_calib_path, velo_calib_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing AV Perception Pipeline on device: {self.device}")

        self.detector = self.load_detector()

        # Load calibration data from both files
        cam_calib = KittiCalibration(cam_calib_path)
        velo_calib = KittiCalibration(velo_calib_path)

        # --- COMBINE MATRICES CORRECTLY ---
        # 1. Get P2 (projection matrix) from the camera calibration file
        self.P2 = cam_calib.get_camera_matrix().reshape(3, 4)

        # 2. Get R0_rect (rectification matrix) from the camera calibration file
        R0_rect = cam_calib.get_rectification_matrix().reshape(3, 3)
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R0_rect

        # 3. Get R and T (LiDAR to Cam) from the velodyne calibration file
        R, T = velo_calib.get_lidar_to_cam_transform()
        R = R.reshape(3, 3)
        T = T.reshape(3, 1)

        # 4. Create the Velodyne to Camera 0 transformation matrix
        Tr_velo_to_cam_4x4 = np.eye(4)
        Tr_velo_to_cam_4x4[:3, :3] = R
        Tr_velo_to_cam_4x4[:3, 3] = T.flatten()

        # 5. Combine all transformations to get the final matrix
        self.T_lidar_to_cam = R0_rect_4x4 @ Tr_velo_to_cam_4x4
        
        print("Pipeline initialized successfully.")

    def load_detector(self):
        print("-> Loading YOLOv8 detector model...")
        model = YOLO('yolov8n.pt')
        model.to(self.device)
        return model
        
    def project_lidar_to_image(self, lidar_points):
        """Projects LiDAR points from 3D to 2D image coordinates."""
        # Keep only points in front of the LiDAR sensor
        lidar_points = lidar_points[lidar_points[:, 0] > 0]
        
        # Convert points to homogeneous coordinates (add a 1)
        points_h = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
        
        # Transform points from LiDAR to camera coordinates
        points_cam = self.T_lidar_to_cam @ points_h.T
        
        # Project points onto the 2D image plane
        points_img = (self.P2 @ points_cam).T
        
        # Normalize by the depth coordinate (z)
        points_img[:, :2] /= points_img[:, 2][:, np.newaxis]
        
        # Return 2D points and their original depths (distance)
        return points_img[:, :2], lidar_points[:, 0] 

    # UPDATED DRAWING FUNCTION
    def draw_fused_detections(self, image, detections, lidar_points_2d, depths):
        """Draws bounding boxes with distance information."""
        h, w, _ = image.shape
        class_names = detections.names

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Find all LiDAR points that fall within this bounding box
            mask = (lidar_points_2d[:, 0] >= x1) & (lidar_points_2d[:, 0] < x2) & \
                   (lidar_points_2d[:, 1] >= y1) & (lidar_points_2d[:, 1] < y2)
            
            points_in_box = depths[mask]
            
            distance = -1
            if len(points_in_box) > 0:
                # Calculate the median distance (more robust to outliers)
                distance = np.median(points_in_box)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label with class, confidence, and distance
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            
            dist_text = f"{distance:.1f}m" if distance != -1 else "N/A"
            label = f"{class_name}: {confidence:.2f} ({dist_text})"
            
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return image


if __name__ == '__main__':
    # Define base directories correctly
    sync_dir = 'data/2011_09_26_drive_0013_sync/'
    calib_dir = 'data/2011_09_26/'
    
    # Define specific data folders
    image_dir = os.path.join(sync_dir, 'image_02/data/')
    lidar_dir = os.path.join(sync_dir, 'velodyne_points/data/')
    
    # Get sorted lists of files
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    lidar_files = sorted([os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if f.endswith('.bin')])

    # --- CORRECTED CODE ---
    # We now load both calibration files
    cam_calib_path = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    velo_calib_path = os.path.join(calib_dir, 'calib_velo_to_cam.txt')

    # Pass both paths to the pipeline
    pipeline = AVPerceptionPipeline(cam_calib_path, velo_calib_path)
    
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        lidar_points = np.fromfile(lidar_files[i], dtype=np.float32).reshape(-1, 4)
        
        detections = pipeline.detector(frame)[0]
        points_2d, depths = pipeline.project_lidar_to_image(lidar_points)
        annotated_frame = pipeline.draw_fused_detections(frame, detections, points_2d, depths)
        
        display_width = 1280
        h, w, _ = annotated_frame.shape
        display_height = int(display_width * (h / w))
        resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

        cv2.imshow("Fused Perception", resized_frame)
        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()