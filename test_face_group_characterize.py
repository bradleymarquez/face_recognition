#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw
import numpy as np
import os
import face_recognition

# Load all images from a given directory
def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = face_recognition.load_image_file(os.path.join(folder,filename))
        if img is not None:
            print "Loaded", filename
            images[os.path.splitext(filename)[0]] = img
        else:
            print "Error: Failed to read file", filename
    return images

imageData = {}
images = load_images_from_folder("fbFaces/qb")

for name, image in images.iteritems():
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            avg_face_landmarks_list = {};
            xTotal = 0;
            yTotal = 0;
            count = 0;
            for point in face_landmarks[facial_feature]:
                xTotal += point[0]
                yTotal += point[1]
                count += 1
            xAvg = xTotal / count;
            yAvg = yTotal / count;
            avg_face_landmarks_list[facial_feature] = (xAvg, yAvg)
            print("The {} in this face has the average location: {}".format(facial_feature, avg_face_landmarks_list[facial_feature]))
        imageData[name] = (image, avg_face_landmarks_list)