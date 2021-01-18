import cv2
# import imutils
import numpy as np
import dlib
 
# creating a list of facial coordinates
def landmarksToCoordines(landmarks, dtype="int"):
    # initializing the list with the 68 coordinates
    coord = np.zeros((68, 2), dtype=dtype)
 
    # go through the 68 coordinates and return
    # coordinates in a list with the format (x, y)
    for i in range(0, 68):
        coord[i] = (landmarks.part(i).x, landmarks.part(i).y)
 
    # return the list of coordinates
    return coord


class Head_Pose_Tracking(object):
    def __init__(self, frame, landmarks):
        self.frame = frame    
        # self.frame = imutils.resize(self.frame, width=400)
        self.landmarks = landmarks
        self.landmarks = landmarksToCoordines(self.landmarks)
        self.key_features=[]
        self.key_features.append(self.landmarks[30])  #tip of the nose
        self.key_features.append(self.landmarks[8])	#chin tip
        self.key_features.append(self.landmarks[36])	#left corner of the eye
        self.key_features.append(self.landmarks[45])	#right corner of the eye
        self.key_features.append(self.landmarks[48])	#left corner of the mouth
        self.key_features.append(self.landmarks[54])	#right corner of the mouth
    
    def analyze(self):
        points_2d = np.asarray(self.key_features, dtype=np.float32).reshape((6, 2))     
        points_3d = np.array([
                            (0.0, 0.0, 0.0),
                            (0.0, -330.0, -65.0),
                            (-225.0, 170.0, -135.0),
                            (225.0, 170.0, -135.0),
                            (-150.0, -150.0, -125.0),
                            (150.0, -150.0, -125.0)
                            ])                    
        dimensions = self.frame.shape
        focal_length = dimensions[1]
        center = (dimensions[1]/2, dimensions[0]/2)
        cam_matrix = np.array(
                                    [[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double"
                                    )                    
        coef_dist = np.zeros((4,1)) # assume we have no camera distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(points_3d, points_2d, cam_matrix, coef_dist, flags=cv2.SOLVEPNP_ITERATIVE)                                    
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, cam_matrix, coef_dist)
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        cv2.putText(self.frame, "X: " + "{:7.2f}".format(euler_angle[1, 0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
        cv2.putText(self.frame, "Y: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
        cv2.putText(self.frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)              
        for p in points_2d:
            cv2.circle(self.frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        p1 = ( int(points_2d[0][0]), int(points_2d[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        # print (points_2d, nose_end_point2D)
        cv2.line(self.frame, p1, p2, (255,0,0), 2)      
        # cv2.imshow('head_angles', self.frame)    

        return (euler_angle[1, 0], euler_angle[0, 0], euler_angle[2, 0])
