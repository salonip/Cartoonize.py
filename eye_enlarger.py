from imutils import face_utils,rotate_bound
import numpy as np
import imutils
import dlib
import cv2
import math
from collections import OrderedDict

class LargeEyes:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.FACIAL_LANDMARKS_IDXS = OrderedDict([("right_eye", (36, 42)),("left_eye", (42, 48))])

        
    def calculate_inclination(self,point1, point2):
        x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
        incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
        return incl
        
    def enlarge(self,image):
        image =imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        for (i, rect) in enumerate(rects):
            (x,y,w,h) = (rect.left(), rect.top(), rect.width(), rect.height())
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            incl = self.calculate_inclination(shape[17], shape[26])
            for (name, (i, j)) in self.FACIAL_LANDMARKS_IDXS.items():
                (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = image[y1:y1 + h1, x1:x1 + w1]
                roi = imutils.resize(roi, width=int(w1+0.10*w1), inter=cv2.INTER_CUBIC)
                (x2,y2,w2,h2) = (x1,y1+2,int(w1+0.10*w1),roi.shape[0])
                image[y1:y1 + h2, x1:x1 + w2] =roi
                image[y1:y1 + h2+50, x1:x1 + w2+50]  = cv2.blur(image[y1:y1 + h2+50, x1:x1 + w2+50] ,(2,2))
        return image
'''       
filename = 'AnnaFrozen.png'
le = LargeEyes()
img = le.enlarge(filename)
print(img)
cv2.imshow('eye',img)
cv2.waitKey(0)

'''


    

