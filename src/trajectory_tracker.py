import os
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from includes.modules import TrajectoryTracking

class TrajectoryTracker(TrajectoryTracking):
    def __init__(self, camera_calib, matches1=None, matches2=None) -> None:
        self.__camera_calib = camera_calib
        self.__matches1 = matches1
        self.__matches2 = matches2
        self.__current_position = np.zeros((3,1))
        self.camera_positions = []
        self.camera_positions.append(self.__current_position)

    def update_trajectory(self, matches1, matches2):
        
        self.__matches1 = np.int32(matches1)
        self.__matches2 = np.int32(matches2)

        if self.__matches1 is None or self.__matches2 is None:
            raise Exception("MAJOR ERROR")

        F, mask = cv.findFundamentalMat(self.__matches1, self.__matches2, cv.FM_8POINT)
        # # Print the fundamental matrix and the number of inliers
        # print("Fundamental matrix:\n", F)
        # print("Number of inliers:", np.sum(mask))
        E, _ = cv.findEssentialMat(self.__matches1, self.__matches1, self.__camera_calib, cv.RANSAC)

        pts, R, t, mask = cv.recoverPose(E, self.__matches1, self.__matches2, self.__camera_calib)        
        self.__current_position = R.dot(self.__current_position) + t
        self.camera_positions.append(self.__current_position)




    def plot_trajectory(self):
        print(self.camera_positions)        

        plots = np.array(self.camera_positions).T

        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(plots[0][0], plots[0][1], '-b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        ax.set_title('Vehicle Trajectory')
        plt.show()


