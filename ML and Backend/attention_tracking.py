import cv2
import dlib
import numpy as np
import pandas as pd
import math

from eye_tracking import Eye_tracking
from mouth_tracking import Mouth_tracking
from gaze_tracking import Gaze_Tracking
# from head_pose_estimation import Pose_Estimation

# cap = cv2.VideoCapture('test_video/test3.mp4')
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("trained_models/shape_predictor_68_face_landmarks.dat")

       
# img=None

# def nothing(x):
#     pass

# cv2.namedWindow('image')
# cv2.createTrackbar('threshold', 'image', 0, 255, nothing)


# def draw_landmark(frame, part_no):
#     cv2.circle(frame,(landmarks.part(part_no).x,landmarks.part(part_no).y),2,(255,0,0),1)
#     cv2.putText(frame, str(part_no), (landmarks.part(part_no).x,landmarks.part(part_no).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)




# left = [36, 37, 38, 39, 40, 41]
# right = [42, 43, 44, 45, 46, 47]
# # kernel = np.ones((9, 9), np.uint8)


# count=0
# cum_l_ratio, cum_r_ratio,cnt=0,0,0
# calibration_count = 0
# blink, yawn, center_focus = 0,0,0
# features = pd.DataFrame(columns=["eye_ratio", "eye_white_ratio", "mouth_ratio"])
# while(True):
#     # Capture frame-by-frame
#     ret, img = cap.read()
#     print(img.shape)
#     if not ret:                                         # When video ends, break out of loop
#         break
#     # Our operations on the frame come here
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Converting colored video to grayscale for efficient processing
    
#     # Basic features that would be mainly used in attention tracking
    
    
#     # gray = cv2.resize(gray,(int(gray.shape[1]*0.8),int(gray.shape[0]*0.8)),interpolation = cv2.INTER_AREA)

#     faces = detector(gray)
#     # First point of detection -
#     # If no faces found in video frame, put alert text and skip over to next frame
#     if not faces:
#         cv2.putText(img, 'NO FACES FOUND !', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 3)

#     for face in faces:
#         # rectangle enclosing the face which will contain 68 facial landmarks
#         cv2.rectangle(gray,(face.left(),face.top()),(face.right(),face.bottom()), (0,255,0), 2)
#         landmarks = predictor(gray,face)
#         gaze_tracker = Gaze_Tracking(gray, landmarks)
#         eye_tracker = Eye_tracking(gray, landmarks)
#         mouth_tracker = Mouth_tracking(gray, landmarks)
#         # pose_tracker = Pose_Estimation(img,landmarks)
#         # landmarks_pts = np.arange(0,68)
#         # for i in landmarks_pts:
#         #     draw_landmark(gray, i)           

#         # left_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left])
#         # right_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right])        


#         if calibration_count < 10:
#             blink += eye_tracker.analyze()
#             yawn += mouth_tracker.analyze()
#             center_focus += gaze_tracker.analyze()
#             calibration_count+=1
#             continue
#         elif calibration_count==10:
#             blink, yawn, center_focus = blink/10,yawn/10,center_focus/10
#             print(blink, yawn, center_focus)
#             calibration_count+=1

#         eye_white_ratio = gaze_tracker.analyze()

#         if eye_white_ratio > 1.4*center_focus :     #1.8:
#             cv2.putText(img, 'LOOKING RIGHT', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
#             print('right',eye_white_ratio)
#         elif eye_white_ratio < 0.7*center_focus:    #0.8 :
#             cv2.putText(img, 'LOOKING LEFT', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
#             print('left',eye_white_ratio)
#         else: 
#             cv2.putText(img, 'LOOKING CENTER', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
#         # print('center',(left_eye_white_ratio+right_eye_white_ratio)/2.0)

#         eye_lengths_ratio = eye_tracker.analyze()
#         # print((l+r)/2.0)
#         if eye_lengths_ratio > 1.3*blink:#3.7: #HARDCODED FOR NOW --- needs to train model, normalize facial features to determine threshold value for each individual
#             cv2.putText(img, 'BLINKING/EYES CLOSED', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
#         # print(l,r, (l+r)/2.0)
        

#         mouth_ratio = mouth_tracker.analyze()
#         if mouth_ratio < 0.75*yawn:#2.0: #HARDCODED FOR NOW --- needs to train model, normalize facial features to determine threshold value for each individual
#             cv2.putText(img, 'YAWNING', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
#         # print(mouth_ratio)

#         # cum_l_ratio = cum_l_ratio + l
#         # cum_r_ratio = cum_r_ratio + r
#         # print(l,r, "\t\t ==>\t", cum_l_ratio, cum_r_ratio)
#         # cnt+=1
#         # print(pose_tracker.analyze())
#         features_series = pd.Series([eye_lengths_ratio, eye_white_ratio, mouth_ratio], index = features.columns)
#         features = features.append(features_series, ignore_index=True)
#         print(features)
#     cv2.imshow('image',img)
#     count+=6
#     cap.set(1,count)
#     cv2.imshow('gray frame',gray)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# # print(cum_l_ratio/cnt)
# # print(cum_r_ratio/cnt)
# # features.to_csv('dset1.csv')
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
gray = cv2.imread('download.webp', 0)
