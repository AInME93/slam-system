import os

import cv2 as cv
import numpy as np

from src.frame_reader import KittiFrameLoader
from src.calib_reader import KittiCalibrationLoader
from src.feature_tracker import FeaturesTracker
from src.feature_match import FeaturesMatcher
from src.trajectory_tracker import TrajectoryTracker

cur_position = np.zeros((3,1))
camera_positions = []
camera_positions.append(cur_position)
kc = KittiCalibrationLoader("data/calibration", "calib_cam_to_cam.txt")
kc.read_calib_file()
K = kc.get_calib_value("K_00", (3,3))



if __name__ == "__main__":
    tt = TrajectoryTracker(K)
    kt = KittiFrameLoader("data/frames/image_02")
    increment = 1

    for i in range(kt.num_frames - increment):
        img = kt.load_frame(i)
        img2 = kt.load_frame(i+increment)
        if img is None or img2 is None:
            continue #current implementation continue, for future: calculate middle coordinates between previous iteration and next and add
        # kt.display_frame(1)
        ft = FeaturesTracker(img)
        ft2 = FeaturesTracker(img2)
        # print(f'Total: {ft.features_detected} for image: {i}')
        # ft.display_kp_img()

        fm = FeaturesMatcher(ft.des, ft2.des, ft.kp, ft2.kp)
        matches1, matches2 = fm.get_matches()
        tt.update_trajectory(matches1, matches2)

    tt.plot_trajectory()
