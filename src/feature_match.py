import os
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import cv2 as cv
import numpy as np

from includes.modules import FeatureMatcher

class FeaturesMatcher(FeatureMatcher):

    def __init__(self, descr1, descr2, kp1, kp2) -> None:
        self.__descr1 = descr1
        self.__descr2 = descr2
        self.__kp1 = kp1
        self.__kp2 = kp2

        self.match_features()
        self.filter_matches()

    def match_features(self):        
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        # self.__matches = bf.match(self.__descr1, self.__descr2)
        self.__matches = bf.knnMatch(self.__descr1, self.__descr2, k=2)

    def get_matches(self):
        matches_img1 = [self.__kp1[m.queryIdx].pt for m in self.__matches]
        matches_img2 = [self.__kp2[m.trainIdx].pt for m in self.__matches]

        return matches_img1, matches_img2

    def filter_matches(self):

        good_matches = []
        # Apply distance ratio test to filter out ambiguous matches
        for m,n in self.__matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        self.__matches = good_matches



