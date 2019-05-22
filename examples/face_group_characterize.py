#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw
import numpy as np
import os
import face_recognition

# Takes in weighted sum of the inputs and normalizes
# them through between 0 and 1 through a sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The derivative of the sigmoid function used to
# calculate necessary weight adjustments
def sigmoid_derivative(x):
    return x * (1 - x)

# Pass inputs through the neural network to get output
def think(inputs, synaptic_weights):
    inputs = inputs.astype(float)
    output = sigmoid(np.dot(inputs, synaptic_weights))
    return output

# We train the model through trial and error, adjusting the
# synaptic weights each time to get a better result
def train(training_inputs, training_outputs, training_iterations):
     # Seed the random number generator
    np.random.seed(1)

    # Set synaptic weights to a 3x1 matrix,
    # with values from -1 to 1 and mean 0
    synaptic_weights = 2 * np.random.random((3, 1)) - 1

    for iteration in range(training_iterations):
        # Pass training set through the neural network
        output = think(training_inputs, synaptic_weights)

        # Calculate the error rate
        error = training_outputs - output

        # Multiply error by input and gradient of the sigmoid function
        # Less confident weights are adjusted more through the nature of the function
        adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))

        # Adjust synaptic weights
        synaptic_weights += adjustments
        
        return synaptic_weights       

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

## imageData[face] -> (image, {facial feature -> (x, y)}
# Get image data for a picture set
def get_picture_data(folder):
    imageData = {}
    images = load_images_from_folder(folder)
    
    for face, image in images.iteritems():
        # Find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(image)
    
        for face_landmarks in face_landmarks_list:
            # Print the location of each facial feature in this image
            for facial_feature in face_landmarks.keys():
                avg_face_landmarks_list = {};
                
                # Find average location of each facial feature
                xTotal = 0;
                yTotal = 0;
                for point in face_landmarks[facial_feature]:
                    xTotal += point[0]
                    yTotal += point[1]
                xAvg = xTotal / len(face_landmarks)
                yAvg = yTotal / len(face_landmarks)
                avg_face_landmarks_list[facial_feature] = (xAvg, yAvg)
                print("The {} for face {} has the average location: {}".format(facial_feature, face, avg_face_landmarks_list[facial_feature]))
    
            imageData[face] = avg_face_landmarks_list
    
    return imageData

def main():
   
    ## imageData[face] -> (image, {facial feature -> (x, y)}
    # input -> {facial feature -> coordinate, .., .., ..}
    # output -> positon (e.g. qb/no-qb)
    
    allFaceData = []
    fbPositions = []
    
    position = "qb" #TODO: iterate through all positions
    imageData = get_picture_data("fbFaces/" + position)
    for face, faceData in imageData.iteritems():
        locationPerFacialFeature = []
        for facialFeature in faceData:
            print faceData[facialFeature]
            # facial feature x-coordinate
            locationPerFacialFeature.append(faceData[facialFeature][0])
            # facial feature y-coordinate
            locationPerFacialFeature.append(faceData[facialFeature][1])
        print locationPerFacialFeature
        allFaceData.append(locationPerFacialFeature)
        fbPositions.append(position)
        
    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
    
    # Train the neural network
    synaptic_weights = train(allFaceData, fbPositions, 10000)
    
    print("Synaptic weights after training: ")
    print(synaptic_weights)
    
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    
    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(think(np.array([A, B, C])))

if __name__ == "__main__":
    main()
            