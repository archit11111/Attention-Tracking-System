import cv2
import dlib 
import numpy as np
import base64

from eye_tracking import Eye_tracking
from mouth_tracking import Mouth_tracking
from gaze_tracking import Gaze_Tracking
from head_pose_estimation import Head_Pose_Tracking

import joblib

class server_image_processing(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("trained_models/shape_predictor_68_face_landmarks.dat")
        self.model = joblib.load('trained_models/KNN.pkl')

    def data_uri_to_cv2_img(self, uri):
        encoded_data = uri.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    '''
    calibration : {eye: double, mouth: double, gaze: double}
    '''
    def process_image(self, data_uri, calibration):
        # cap = cv2.VideoCapture('test_video/test3.mp4')
        img = self.data_uri_to_cv2_img(data_uri)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = self.detector(gray)
        if not faces:
            return ({'success':False, 'error': 'No Faces Detected'},{'isCalibrating':None, 'isAttentive':None}, calibration )
        
        face = faces[0]
        landmarks = self.predictor(gray,face)
        eye_tracker = Eye_tracking(gray, landmarks)
        gaze_tracker = Gaze_Tracking(gray, landmarks)
        mouth_tracker = Mouth_tracking(gray, landmarks)        
        head_pose_tracker = Head_Pose_Tracking(gray, landmarks)

        if calibration['calibration_count'] < 10:
            calibration['eye'] += eye_tracker.analyze()
            calibration['mouth'] += mouth_tracker.analyze()
            calibration['gaze'] += gaze_tracker.analyze()
            calibration['calibration_count'] += 1
            return ({'success':True}, {'isCalibrating':True, 'isAttentive':None}, calibration )
        elif calibration['calibration_count'] == 10:
            calibration['eye'] /= 10
            calibration['mouth'] /= 10
            calibration['gaze'] /= 10
            calibration['calibration_count'] += 1
            
        eye_lengths_ratio = eye_tracker.analyze()
        eye_white_ratio = gaze_tracker.analyze()
        mouth_ratio = mouth_tracker.analyze()
        (theta_X,_,_) = head_pose_tracker.analyze()

        X_test = [(eye_lengths_ratio, eye_white_ratio, mouth_ratio, theta_X)]
        y_out = self.model.predict(X_test)

        if y_out == 0 or eye_lengths_ratio > 1.3*calibration['eye'] or eye_white_ratio > 3*calibration['gaze'] or eye_white_ratio < 0.3*calibration['gaze'] or mouth_ratio < 0.8*calibration['mouth']:
            return ({'success':True}, {'isCalibrating':False, 'isAttentive':False}, calibration )
        else:
            return ({'success':True}, {'isCalibrating':False, 'isAttentive':True}, calibration )
