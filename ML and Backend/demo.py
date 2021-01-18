import cv2
import dlib

from eye_tracking import Eye_tracking
from mouth_tracking import Mouth_tracking
from gaze_tracking import Gaze_Tracking
from head_pose_estimation import Head_Pose_Tracking

import joblib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("trained_models/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
model = joblib.load('trained_models/KNN.pkl')

calibration_count = 0
blink, yawn, center_focus = 0,0,0

while(True):
    
    ret, img = cap.read()
    
    if not ret:                                       
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # First point of detection -
    # If no faces found in video frame, put alert text and skip over to next frame
    if not faces:
        cv2.putText(img, 'NO FACES FOUND !', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 3)    
        cv2.imshow('image frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # take only first face that's detected
    face = faces[0]

    cv2.rectangle(gray,(face.left(),face.top()),(face.right(),face.bottom()), (0,255,0), 2)
    landmarks = predictor(gray,face)

    eye_tracker = Eye_tracking(gray, landmarks)
    gaze_tracker = Gaze_Tracking(gray, landmarks)
    mouth_tracker = Mouth_tracking(gray, landmarks)        
    head_pose_tracker = Head_Pose_Tracking(gray, landmarks)

    if calibration_count < 10:
        blink += eye_tracker.analyze()
        center_focus += gaze_tracker.analyze()
        yawn += mouth_tracker.analyze()
        calibration_count+=1
        continue
    elif calibration_count==10:
        blink, yawn, center_focus = blink/calibration_count,yawn/calibration_count,center_focus/calibration_count
        print(blink, yawn, center_focus)
        calibration_count+=1


    eye_lengths_ratio = eye_tracker.analyze()
    eye_white_ratio = gaze_tracker.analyze()
    mouth_ratio = mouth_tracker.analyze()
    (theta_X,_,_) = head_pose_tracker.analyze()

    X_test = [(eye_lengths_ratio, eye_white_ratio, mouth_ratio, theta_X)]

    
    y_out = model.predict(X_test)
    # cv2.putText(img, str(eye_white_ratio), (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
    cv2.imshow('gray',gray)

    if y_out == 0 or eye_lengths_ratio > 1.3*blink or eye_white_ratio > 2.5*center_focus or eye_white_ratio < 0.35*center_focus or mouth_ratio < 0.65*yawn:
        cv2.putText(img, 'INATTENTIVE', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    else:
        cv2.putText(img, 'ATTENTIVE', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3) 
    
    if eye_lengths_ratio > 1.3*blink:
        cv2.putText(img, 'Sleepy', (50,100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 3)
    if  eye_white_ratio > 2.5*center_focus or eye_white_ratio < 0.35*center_focus:
        cv2.putText(img, 'Not focused on Screen', (50,150), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 3)
    if mouth_ratio < 0.65*yawn:
        cv2.putText(img, 'Yawning', (50,200), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 3)
        

    cv2.imshow('image frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
