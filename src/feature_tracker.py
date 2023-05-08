import os
from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import cv2 as cv

from includes.modules import FeatureTracker
from src.utils import ExtractType

class FeaturesTracker(FeatureTracker):
    def __init__(self, img, n: int = 5000, track_type: ExtractType = ExtractType.ORB) -> None:
        self.__img__ = img
        self.__nfeatures__ = self.__threshold__ = n
        self.__track_type__ = track_type
        self.kp_img = None
        self.features_detected = None

        if self.__img__ is None:
            raise Exception("Error capturing frame data.")

        if self.__track_type__ == ExtractType.ORB:
            self.orb_extract_features()
        else:
            self.fast_extract_features()
        # self.set_kp_img()

    def orb_extract_features(self):
        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures = self.__nfeatures__)

        # compute the descriptors with ORB
        self.kp, self.des = orb.detectAndCompute(self.__img__, None)
        self.features_detected = len(self.kp)

    def fast_extract_features(self):

        # Initiate FAST object with default values
        fast = cv.FastFeatureDetector_create()

        # find and draw the keypoints
        self.kp = fast.detect(self.__img__,None)

        # sort features based on response value
        self.kp = sorted(self.kp, key=lambda x: -x.response)[:200]

        img2 = cv.drawKeypoints(self.__img__, self.kp, None, color=(255,0,0))

        # Print all default params
        fast.setThreshold(self.__threshold__)
        print( "Total Keypoints with nonmaxSuppression: {}".format(len(self.kp)) )
        cv.imwrite('fast_true.png', img2)
        cv.imshow("With NonMaxSuppression", img2)

        # Disable nonmaxSuppression
        fast.setNonmaxSuppression(0)
        self.kp = fast.detect(self.__img__, None)
        print( "Total Keypoints without nonmaxSuppression: {}".format(len(self.kp)) )
        img3 = cv.drawKeypoints(self.__img__, self.kp, None, color=(255,0,0))
        
        cv.imshow("Without NonMaxSuppression", img3)

    def set_kp_img(self):
        self.kp_img = cv.drawKeypoints(self.__img__, self.kp, None, color=(255,0,0))

    def display_kp_img(self):
        cv.imshow("Keypoints", self.kp_img)   
        cv.waitKey()
        cv.destroyAllWindows()
    
    # def sort_matches(self): 
    #     matches = sorted(matches)        

