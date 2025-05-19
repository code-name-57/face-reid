####################################
# LIBRARIES
from deepface import DeepFace
import cv2
import time
import matplotlib.pyplot as plt
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
# GO THROUGH VIDEO
frameNum = 1
faceNum = 0
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True: # if there is a frame to read
        cv2.imshow('Frame', frame)

        # EXTRACT & SAVE FACES FROM FRAME
        try:
            faces = DeepFace.extract_faces( frame )
            for i, face_dict in enumerate(faces): # loop through ever dictionary in list
                faceNum += 1
                print("Face #", faceNum, " in Frame Number", frameNum, "Detected!")
                cv2.imwrite(r"C:\Users\kanie\face-reid\VideoCaptureTask\FacesInFrames\Face" + str(faceNum) + ".jpg", frame)
        except:
            print("No Faces Detected")

        frameNum += 1
        k = cv2.waitKey(20)
        cv2.imwrite(r"C:\Users\kanie\face-reid\VideoCaptureTask\AllFrames\Frame" + str(frameNum) + ".jpg", frame)

        # For exiting video early
        if k == 113: 
            break # break loop if q pressed (waitkey linked to q)
    else:
        break

video.release()
cv2.destroyAllWindows()
####################################



####################################
# PROGRESS
# 
# Currently:
# -> Process a video / go through a video frame-by-frame
# -> Extract faces from each frame
# -> Save to a faces folder
# 
# To-Do:
# -> Get a hold of lower threshold limits (e.g. use deepface on saved images)
# -> Be selective about faces (really use Deepface)
#   -> Use face recognition/verify on each frame
#   -> Only save it if within upper + lower threshold
#
# In Future
#   -> Have a sub-folder for each  person??
#
####################################
