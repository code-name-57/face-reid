####################################
# LIBRARIES
from deepface import DeepFace
import cv2
import time
import matplotlib.pyplot as plt
####################################

####################################
# TESTING STUFF

imgTest = r"C:\Users\kanie\face-reid\VideoCaptureTask\img1.jpg"

try:
    faces = DeepFace.extract_faces( imgTest )
    for i, face_dict in enumerate(faces): # loop through ever dictionary in list
        plt.imshow(face_dict["face"]) # show the "face" of the current face_dict
        plt.title(f"Face {i+1}")  # add title to face plot
        plt.show()
except:
    print("No Faces Detected")
