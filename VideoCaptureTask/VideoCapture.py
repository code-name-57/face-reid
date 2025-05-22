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
# REMOVE ANY SAVED DATA
# (can comment out if needed)
import shutil 
pathToDelete = r"C:\Users\kanie\face-reid\VideoCaptureTask\People"
shutil.rmtree( pathToDelete )
ReCreatePath = r"C:\Users\kanie\face-reid\VideoCaptureTask\People"
os.makedirs( ReCreatePath )
####################################


####################################
# OPEN VIDEO
video = cv2.VideoCapture(r"C:\Users\kanie\face-reid\VideoCaptureTask\B99Clip.mp4") 
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

# Changeable
ppl_folder = r"C:\Users\kanie\face-reid\VideoCaptureTask\People"
lower_threshold = 0.12

# Pre-Defined
frameNum = 1            # Total Number of Frames
faceNum = 0             # Total Number of Faces From All Frames
uniquePersonNum = 0     # Total Number of Unique Faces
appearancesArray = []   # Number of Appearances from Each Unique Face
appearancesSaved = []   # Number of Apearances Saved of Each Unique Face
newPerson = True    # Boolean used to check whether a new unique person has been detected
finishCompare = False

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
                finishCompare = False
                faceNum += 1
                print("Face #", faceNum, " in Frame Number", frameNum, "Detected!")
                imageToSave = face_dict["face"]
                # Ensure it's in the right format for saving
                if imageToSave.dtype == np.float32 or imageToSave.max() <= 1.0:
                    imageToSave = (imageToSave * 255).astype(np.uint8)

                # cv2.imwrite(r"C:\Users\kanie\face-reid\VideoCaptureTask\FacesInFrames\Face" + str(faceNum) + ".jpg", frame)

                ####################################
                # GO THROUGH FILE SYSTEM
                    # Note: commented out print-statements to save execution time but left in just in case

                # Iterate over folders in directory
                for personNum in os.listdir(ppl_folder):

                    # Check if can skip ahead (based on prev. loop-runs)
                    if finishCompare == True:
                        break

                    # Get Path of PersonX subfolder
                    person_folder = os.path.join(ppl_folder, str(personNum) )
                    print(f"Reading Folder: {person_folder}")

                    if os.path.isdir(person_folder):    # to confirm looking at a folder, not some file
                        images = os.listdir(person_folder)  # get all images in folder in array
                        toAddImage = False

                        # Loop through images in folder for comparisions
                        print("Comparing...")
                        for x in range(len(images)):
                            print("Comparing w Image ", str(x+1))
                            image_path = os.path.join(person_folder, images[x])

                            # Use OpenCV to read & display images
                            img_in_folder = cv2.imread( image_path )
                        
                            compare = DeepFace.verify( img_in_folder, imageToSave )
                            # COMPARISIONS
                            # Check if Same Person as image in folder
                            if compare['distance'] < compare['threshold']:
                                newPerson = False
                                # Check if below lower-threshold (aka too similar to save)
                                if compare['distance'] < lower_threshold:
                                    print("Person Detected BUT Below Lower Threshold!")
                                    print()
                                    finishCompare = True
                                    toAddImage = False
                                    break
                                # Within upper & lower thresholds amd compared w other images in folder too
                                else: 
                                    toAddImage = True
                            else:
                                print("Moving onto next image!")
                                print()
                        
                        if toAddImage == True:
                            # Add Image to Folder
                            newImagePath = person_folder + r"\Face" + str(faceNum) + ".jpg"
                            cv2.imwrite( newImagePath, imageToSave)
                            print("Added Person to Existing Folder!")
                        else:
                            print("Person not added to Folder!")
                        
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
# -> Get a hold of lower threshold limits (e.g. use deepface on saved images)
# -> Be selective about faces (really use Deepface)
#   -> Use face recognition/verify on each frame
#   -> Only save it if within upper + lower threshold
# 
# In The Future (for this specific version):
# -> Re-write cleaner and more concise/shorter
# -> Add an extra layer of checking for valid images
#   -> e.g. what happened w the B99 clip
# -> Increase Accuracy (not just HAC)
# -> Re-try to the SkipThreshold
# -> Decrease execution time
#   -> e.g. compare w previous frames/faces first (which is more likely to be similar)
#   -> tweak the thresholds more
# -> Use embeddings instead of raw images (for privacy and ease)
# -> STATS!! (in nice visuals, use matplotlib.py ?)
#
####################################
