####################################
# LIBRARIES
from deepface import DeepFace
import cv2
import time
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
####################################

####################################
# OPEN VIDEO
video = cv2.VideoCapture(r"C:\Users\kanie\face-reid\VideoCaptureTask\HomeAloneClip.mp4") 
    # need to add 'r' to ensure backslash isn't used as escape character


# Confirm if video opened successfully (get video info if so)
if( video.isOpened() == False ):
    print("Error opening the video file")
else:
    # Get frame rate information ('5' represents that command)
    fps = int( video.get(5) )
    print("Frame Rate : ", fps, "frames per second")

    # Get frame count ('7' represents that command)
    frame_count = video.get(7)
    print("Frame Count : ", frame_count)

####################################
# VARIABLE DEFINITIONS
frameNum = 1            # Total Number of Frames
faceNum = 0             # Total Number of Faces From All Frames
uniquePersonNum = 0     # Total Number of Unique Faces
appearancesArray = []   # Number of Appearances from Each Unique Face
appearancesSaved = []   # Number of Apearances Saved of Each Unique Face

ppl_folder = r"C:\Users\kanie\face-reid\VideoCaptureTask\People"
newPerson = True    # Boolean used to check whether a new unique person has been detected

####################################
# GO THROUGH VIDEO
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True: # if there is a frame to read
        cv2.imshow('Frame', frame)

        ####################################
        # EXTRACT & SAVE FACES FROM FRAME
        try:
            faces = DeepFace.extract_faces( frame )
            for i, face_dict in enumerate(faces): # loop through ever dictionary in list
                newPerson = True
                faceNum += 1
                print("Face #", faceNum, " in Frame Number", frameNum, "Detected!")
                imageToSave = face_dict["face"]
                # Ensure it's in the right format for saving
                if imageToSave.dtype == np.float32 or imageToSave.max() <= 1.0:
                    imageToSave = (imageToSave * 255).astype(np.uint8)

                # cv2.imwrite(r"C:\Users\kanie\face-reid\VideoCaptureTask\FacesInFrames\Face" + str(faceNum) + ".jpg", frame)

                ####################################
                # GO THROUGH FILE SYSTEM

                # Iterate over folders in directory
                for personNum in os.listdir(ppl_folder):
                    # Get Path of PersonX subfolder
                    person_folder = os.path.join(ppl_folder, str(personNum) )
                    if os.path.isdir(person_folder):
                        images = os.listdir(person_folder)
                        first_img = images[0]
                        image_path = os.path.join(person_folder, first_img)

                        # Use OpenCV to read & display images
                        print(f"Reading Folder: {image_path}")
                        img_in_folder = cv2.imread( image_path )

                        print("Comparing...")
                        compare = DeepFace.verify( img_in_folder, imageToSave )
                        # Check if same person
                        if compare['distance'] < compare['threshold']:
                            # Add Image to Folder for now (deal w lower threshold after)
                            newImagePath = person_folder + r"\Face" + str(faceNum) + ".jpg"
                            cv2.imwrite( newImagePath, imageToSave)
                            print("Added Person to Existing Folder!")
                            print("")
                            newPerson = False
                            break
                        else:
                            print("Person not added to Folder!")
                            print("Moving onto next folder!")
                            print("")
                        
                # Need to Add New Folder for New Person
                if newPerson == True:
                    uniquePersonNum += 1
                    newFolderPath = ppl_folder + r"\Person" + str(uniquePersonNum)
                    os.makedirs( newFolderPath )
                    print(f"Directory '{newFolderPath}' created successfully.")
                    newImagePath = newFolderPath + r"\Face" + str(faceNum) + ".jpg"
                    cv2.imwrite( newImagePath, imageToSave)
                else:
                    print("Finished with No New Folders Created!")

        except:
            print("No Faces Detected")

        frameNum += 1
        k = cv2.waitKey(20)
        # cv2.imwrite(r"C:\Users\kanie\face-reid\VideoCaptureTask\AllFrames\Frame" + str(frameNum) + ".jpg", frame)

        # For exiting video early
        if k == 113: 
            break # break loop if q pressed (waitkey linked to q)
    else:
        break

video.release()
cv2.destroyAllWindows()

# End Stats
print()
print("******COMPLETED VIDEO SCAN******")
print("Total Frames: ", frameNum)
print("Total Faces Detected: ", faceNum)
print("Total Unique Faces: ", uniquePersonNum)
####################################



####################################
# PROGRESS
# 
# Currently:
# -> Process a video / go through a video frame-by-frame
# -> Extract faces from each frame
# -> Save to a faces folder
# -> Have a sub-folder for each  person
# 
# To-Do:
# -> Get a hold of lower threshold limits (e.g. use deepface on saved images)
# -> Be selective about faces (really use Deepface)
#   -> Use face recognition/verify on each frame
#   -> Only save it if within upper + lower threshold
#
# In Future
# -> Save faces in a proper database instead
# -> Use embeddings instead of raw images (for privacy)
#
####################################
