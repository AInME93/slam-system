import os

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import cv2 as cv

from includes.modules import FrameLoader
from src.utils import set_np_arr

class KittiFrameLoader(FrameLoader):
    def __init__(self, dir_path):
        self.directory = os.path.realpath(os.path.join(os.getcwd(), dir_path))
        self.__files__ = os.listdir(self.directory)
        self.num_frames = len([f for f in self.__files__ if f.endswith('.png')])

        if self.num_frames == 0:
            raise Exception("No frames present in specified folder")

        print(f"Detected {self.num_frames} frames in folder")

    def load_frame(self, file_idx):
        if file_idx > self.num_frames:
            raise Exception("Reached end of files")
        
        frame = cv.imread(os.path.join(self.directory, self.__files__[file_idx]), cv.IMREAD_GRAYSCALE)
        return frame
    
    def get_dir(self):
        print(self.directory)
    
    def display_frame(self, file_idx):
        img = self.load_frame(file_idx)
        cv.imshow(str(file_idx),img)
        cv.waitKey()
        cv.destroyAllWindows()
            
            
if __name__ == "__main__":
    kt = KittiFrameLoader("data/frames")
    kt.print_dir()
