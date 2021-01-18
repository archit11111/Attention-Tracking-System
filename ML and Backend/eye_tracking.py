import cv2
import numpy as np
import math

class Eye_tracking(object):
    def __init__(self, frame, landmarks) -> None:
        self.frame = frame
        self.landmarks = landmarks

    def get_horizontal_eye_length(self, left=True):
        landmark_points = [0]*2
        if left == True:
            landmark_points = [36, 39]
        else:
            landmark_points = [42, 45]

        
        left_eye_point = (self.landmarks.part(landmark_points[0]).x,self.landmarks.part(landmark_points[0]).y)
        right_eye_point = (self.landmarks.part(landmark_points[1]).x,self.landmarks.part(landmark_points[1]).y)
        # cv2.line(self.frame, left_eye_point,right_eye_point, (0,0,255),2)
        horizontal_length = math.hypot(right_eye_point[0] - left_eye_point[0], right_eye_point[1] - left_eye_point[1])
        
        return horizontal_length


    def get_vertical_eye_length(self, left=True):
        landmark_points = [0]*4
        if left == True:
            landmark_points=[37, 38, 40, 41]
        else:
            landmark_points=[43, 44, 46, 47]
        
        top_eye_point = ((self.landmarks.part(landmark_points[0]).x + self.landmarks.part(landmark_points[1]).x)//2,(self.landmarks.part(landmark_points[0]).y + self.landmarks.part(landmark_points[1]).y)//2)
        bottom_eye_point = ((self.landmarks.part(landmark_points[2]).x + self.landmarks.part(landmark_points[3]).x)//2,(self.landmarks.part(landmark_points[2]).y + self.landmarks.part(landmark_points[3]).y)//2)
        # cv2.line(self.frame, top_eye_point,bottom_eye_point, (0,0,255),2)
        vertical_length = math.hypot((top_eye_point[0] - bottom_eye_point[0]),(top_eye_point[1] - bottom_eye_point[1]))

        return vertical_length


    def analyze(self):
        left_eye_ratio = self.get_horizontal_eye_length(True)/self.get_vertical_eye_length(True)
        right_eye_ratio = self.get_horizontal_eye_length(False)/self.get_vertical_eye_length(False)
        # print(self.get_horizontal_eye_length(True),self.get_vertical_eye_length(True),'   |   ',self.get_horizontal_eye_length(False),self.get_vertical_eye_length(False))
        return (left_eye_ratio + right_eye_ratio)/2
    

