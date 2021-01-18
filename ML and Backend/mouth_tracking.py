import cv2
import numpy as np
import math

class Mouth_tracking(object):
    def __init__(self, frame, landmarks) -> None:
        self.frame = frame
        self.landmarks = landmarks

    def get_horizontal_mouth_length(self):
        landmark_points = [48,54];
        
        left_mount_point = (self.landmarks.part(landmark_points[0]).x,self.landmarks.part(landmark_points[0]).y)
        right_mount_point = (self.landmarks.part(landmark_points[1]).x,self.landmarks.part(landmark_points[1]).y)
        # cv2.line(self.frame, left_mount_point,right_mount_point, (0,0,255),2)
        horizontal_length = math.hypot(right_mount_point[0] - left_mount_point[0], right_mount_point[1] - left_mount_point[1])

        return horizontal_length


    def get_vertical_mouth_length(self):
        landmark_points = [57,51];
        
        top_mount_point = (self.landmarks.part(landmark_points[0]).x,self.landmarks.part(landmark_points[0]).y)
        bottom_mount_point = (self.landmarks.part(landmark_points[1]).x,self.landmarks.part(landmark_points[1]).y)
        # cv2.line(self.frame, top_mount_point,bottom_mount_point, (0,0,255),2)
        horizontal_length = math.hypot(bottom_mount_point[0] - top_mount_point[0], bottom_mount_point[1] - top_mount_point[1])

        return horizontal_length    


    def analyze(self):
        mouth_ratio = self.get_horizontal_mouth_length()/self.get_vertical_mouth_length()    
        # print(mouth_ratio)
        # cv2.putText(self.frame, str(mouth_ratio), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        return mouth_ratio
