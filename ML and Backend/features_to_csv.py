import cv2
import dlib
import numpy as np
import pandas as pd
import math
import os

from eye_tracking import Eye_tracking
from mouth_tracking import Mouth_tracking
from gaze_tracking import Gaze_Tracking
from head_pose_estimation import Head_Pose_Tracking
from logistic_regression_model import LRModel

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def draw_landmark(frame, part_no):
    cv2.circle(frame,(landmarks.part(part_no).x,landmarks.part(part_no).y),2,(255,0,0),1)
    cv2.putText(frame, str(part_no), (landmarks.part(part_no).x,landmarks.part(part_no).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


features = pd.DataFrame(columns=["file_name", "eye_ratio", "eye_white_ratio", "mouth_ratio", "X"])
folder = 'Att_DataSet/att'
for filename in os.listdir(folder):
# for i in img_names:
    gray = cv2.imread(os.path.join(folder,filename),0)
    if gray is None :
        print(filename)
        continue
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    faces = detector(gray,0)
        # First point of detection -
        # If no faces found in video frame, put alert text and skip over to next frame
    
    if len(faces) == 0:
        print('no face')
        continue
    print('face detected')
    face = faces[0]

     
    landmarks = predictor(gray,face)        # 68 landmarks points provided by dlib
    landmarks_pts = np.arange(0,68)         #  landmark points array

    '''
        Initialising all the feature extraction modules and 
        analysing the face to extract useful data from the frame.
    '''
    gaze_tracker = Gaze_Tracking(gray, landmarks)
    eye_tracker = Eye_tracking(gray, landmarks)
    mouth_tracker = Mouth_tracking(gray, landmarks)
    head_pose_tracker = Head_Pose_Tracking(gray, landmarks)

    eye_lengths_ratio = eye_tracker.analyze()
    eye_white_ratio = gaze_tracker.analyze()
    print(eye_white_ratio)
    mouth_ratio = mouth_tracker.analyze()
    (X,Y,_) = head_pose_tracker.analyze()
    for i in landmarks_pts[36:48]:
        draw_landmark(gray, i) 
    cv2.imshow('image',gray)
    
    features_series = pd.Series([filename, eye_lengths_ratio, eye_white_ratio, mouth_ratio, X], index = features.columns)
    features = features.append(features_series, ignore_index=True)
    # print(features)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



features.to_csv('dset1.csv')
