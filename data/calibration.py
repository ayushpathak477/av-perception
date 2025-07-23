# data/calibration.py
import numpy as np

class KittiCalibration:
    def __init__(self, calib_filepath):
        self.calib_data = self._load_calibration(calib_filepath)

    def _load_calibration(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        data[key] = np.array([float(x) for x in value.split()])
                    except ValueError:
                        data[key] = value
        return data

    def get_camera_matrix(self):
        # Projection matrix for the left color camera (P_rect_02)
        return self.calib_data.get('P_rect_02')

    def get_rectification_matrix(self):
        # Rectification matrix for the reference camera (R_rect_00)
        return self.calib_data.get('R_rect_00')

    def get_lidar_to_cam_transform(self):
        # Transformation from Velodyne to reference camera (R and T)
        r_matrix = self.calib_data.get('R')
        t_vector = self.calib_data.get('T')
        return r_matrix, t_vector