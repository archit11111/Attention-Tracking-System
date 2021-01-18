import cv2
import numpy as np
import math
from cv2 import FONT_HERSHEY_SIMPLEX
from calibration import Calibration

class Gaze_Tracking(object):
    LEFT = [36, 37, 38, 39, 40, 41]
    RIGHT = [42, 43, 44, 45, 46, 47]

    def __init__(self, frame, landmarks) -> None:
        self.frame = frame    
        self.landmarks = landmarks
        self.calibration = Calibration()
    
    def get_eye_gaze(self, left=False):
        eye_region=None
        if left == True:
            eye_region = np.array([(self.landmarks.part(i).x, self.landmarks.part(i).y) for i in self.LEFT])
        else:
            eye_region = np.array([(self.landmarks.part(i).x, self.landmarks.part(i).y) for i in self.RIGHT])        
        mask = np.zeros(self.frame.shape[:2],np.uint8)
        cv2.polylines(mask,[eye_region], True, 255,2)
        cv2.fillPoly(mask,[eye_region],255)
        mask= cv2.bitwise_and(self.frame,self.frame,mask=mask)

        eye_frame_region = self.frame[np.min(eye_region[:,1]): np.max(eye_region[:,1]),np.min(eye_region[:,0]): np.max(eye_region[:,0])]
        if not self.calibration.is_complete():
            self.calibration.evaluate(eye_frame_region, left)

        threshold = self.calibration.threshold(left)
        # cv2.putText(self.frame, str(threshold), (50,200), FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
        # threshold = cv2.getTrackbarPos('threshold', 'image')

        # if threshold==0:
        #     threshold=55
        kernel = np.ones((3, 3), np.uint8)
        eye_frame_region = cv2.bilateralFilter(eye_frame_region, 10, 15, 15)
        eye_frame_region = cv2.erode(eye_frame_region, kernel, iterations=3)
        _, threshold_eye = cv2.threshold(eye_frame_region, threshold, 255, cv2.THRESH_BINARY)
        h,w = threshold_eye.shape
        
        left_half_white = cv2.countNonZero(threshold_eye[0:h, 0:w//2])
        right_half_white = cv2.countNonZero(threshold_eye[0:h, w//2:w])
        total_pixel = np.sum(threshold_eye!=None)
        # print(left_half_white,right_half_white,left_half_white+right_half_white, total_pixel, '     ')
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        left_to_right_ratio = None
        if right_half_white==0 or (left_half_white+right_half_white)==total_pixel:
            left_to_right_ratio=10
        else:            
            left_to_right_ratio=left_half_white/right_half_white

        if left==True:
            # cv2.putText(self.frame, str(left_to_right_ratio), (50,100), FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
            # cv2.putText(self.frame, str(right_half_white), (50,150), FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
            cv2.imshow('thresh_left',threshold_eye)        
            pass
        else:
            # cv2.putText(self.frame, str(left_to_right_ratio), (50,200), FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
            # cv2.putText(self.frame, str(right_half_white), (350,150), FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
            cv2.imshow('thresh_right',threshold_eye)   
            pass
        
        eye_frame_region = cv2.resize(eye_frame_region,None,fx=10,fy=10)
        # cv2.imshow('left_eye_frame_region',left_eye_frame_region)        
        # cv2.imshow('thresh',threshold_eye)        
        # cv2.imshow('mask',mask)                
        return left_to_right_ratio

    def analyze(self):
        return (self.get_eye_gaze(True)+self.get_eye_gaze(False))/2
