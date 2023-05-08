import os
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import cv2 as cv

from includes.modules import CalibrationLoader
from src.utils import set_key_value, set_np_arr

class KittiCalibrationLoader(CalibrationLoader):
    def __init__(self, dir_path: str, file_name: str):
        self.directory = os.path.realpath(os.path.join(os.getcwd(), dir_path))
        self.file_name = file_name
        self.calib_file = os.path.join(self.directory, self.file_name)
    
    def read_calib_file(self):
        self.calib = set_key_value(self.calib_file)

    def get_calib_value(self, calib_key: str, np_shape: tuple):
        calib_str = self.calib[calib_key].strip().split(" ")
        calib_value = set_np_arr(np_shape, calib_str)
        return calib_value 


if __name__ == "__main__":
    kcl = KittiCalibrationLoader("data/calibration", "calib_cam_to_cam.txt")
    kcl.read_calib_file()
    K = kcl.get_calib_value("K_00", (3,3))
    print(K)