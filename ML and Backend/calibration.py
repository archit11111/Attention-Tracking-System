from __future__ import division
import cv2
import numpy as np


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, left):
        """Returns the threshold value for the given eye.

        Argument:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if left == True:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif left == False:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Argument:
            frame (numpy.ndarray): Binarized iris frame
        """
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        if(nb_pixels==0):
            return 0
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """Calculates the optimal threshold to binarize the
        frame for the given eye.

        Argument:
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """
        average_iris_size = 0.5
        trials = {}

        for threshold in range(5, 100, 3):            
            kernel = np.ones((3, 3), np.uint8)
            new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
            new_frame = cv2.erode(new_frame, kernel, iterations=3)
            new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
            iris_frame=new_frame
            trials[threshold] = Calibration.iris_size(iris_frame)
            # print('threshold - ',threshold)
            # print('iris_size -', trials[threshold])

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, left):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if(left==True):
            cv2.imshow('calib_left',eye_frame)
        else:            
            cv2.imshow('calib_right',eye_frame)
        threshold = self.find_best_threshold(eye_frame)

        if left == True:
            self.thresholds_left.append(threshold)
        else:
            self.thresholds_right.append(threshold)
