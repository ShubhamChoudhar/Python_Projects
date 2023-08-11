'''
Created on 11-Mar-2019

@author: harshan
'''

import cv2
import numpy as np
import os 

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier("/home/harshan/Desktop/FaceRecognition-master/HaarCascade/haarcascade_frontalface_default.xml")
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)
    
    return faces,gray_img