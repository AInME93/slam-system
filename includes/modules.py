import os

import numpy as np
from math import isnan
import cv2 as cv

class FrameLoader:
    def __init__(self):
        pass

    def load_frame():
        pass

    def get_dir():
        pass


class CalibrationLoader:
    
    def __init__(self) -> None:
        pass

    def read_calib_file(self):
        pass

    def get_calib_value(self, calib_key: str, np_shape: tuple):
        pass

class FeatureTracker:

    def __init__(self) -> None:
        pass
    
    def orb_extract_features(self):
        pass

    def fast_extract_features(self):
        pass

class FeatureMatcher:
    def __init__(self) -> None:
        pass
        
    def match_features(self):
        pass

    def filter_matches(self):
        '''
        Filter matches by applying distance ratio test
        '''
        pass

class TrajectoryTracking:

    def __init__(self) -> None:
        pass

    def update_trajectory(self):
        pass

    def plot_trajectory(self):
        pass
