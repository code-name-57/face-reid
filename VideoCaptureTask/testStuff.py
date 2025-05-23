####################################
# LIBRARIES
from deepface import DeepFace
import cv2
import time
import matplotlib.pyplot as plt
import os
####################################

####################################
# TESTING STUFF

# ppl_folder = r"C:\Users\kanie\face-reid\VideoCaptureTask\People"
# newPerson = True

# imgExistingPerson = r"C:\Users\kanie\face-reid\VideoCaptureTask\FacesInFrames\Face59.jpg"
# imgNewPerson = r"C:\Users\kanie\face-reid\VideoCaptureTask\FacesInFrames\Face85.jpg"

# imgTest = imgNewPerson # CURRENTLY USING
# imageToSave = cv2.imread( imgTest )

# # Iterate over files in directory
# for personNum in os.listdir(ppl_folder):
#     # Get Path of PersonX subfolder
#     person_folder = os.path.join(ppl_folder, str(personNum) )

#     if os.path.isdir(person_folder):
#         images = os.listdir(person_folder)
#         first_img = images[0]
#         image_path = os.path.join(person_folder, first_img)

#         # Use OpenCV to read & display images
#         print(f"Reading Folder: {image_path}")
#         img_in_folder = cv2.imread( image_path )

#         compare = DeepFace.verify( img_in_folder, imgTest )
#         # Check if same person
#         if compare['distance'] < compare['threshold']:
#             # Add Image to Folder for now (deal w lower threshold after)
#             newImagePath = person_folder + r"\NewImage" + ".jpg"
#             cv2.imwrite( newImagePath, imageToSave)
#             print("Added Person to Existing Folder!")
#             print("")
#             newPerson = False
#             break
#         else:
#             print("Person not added to Folder!")
#             print("Moving onto next folder!")
#             print("")

# # Need to Add New Folder for New Person
# if newPerson == True:
#     newFolderPath = ppl_folder + r"\PersonX"
#     os.makedirs( newFolderPath )
#     print(f"Directory '{newFolderPath}' created successfully.")
#     newImagePath = newFolderPath + r"\NewImage" + ".jpg"
#     cv2.imwrite( newImagePath, imageToSave)
#     newPerson = False
# else:
#     print("Finished with No New Folders Created!")

        # cv2.imshow( "Person" , img )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

imgTest1 = r"C:\Users\kanie\face-reid\VideoCaptureTask\People\Person9\Face68.jpg"
imgTest2 = r"C:\Users\kanie\face-reid\VideoCaptureTask\People\Person11\Face95.jpg"
compare = DeepFace.verify( imgTest1, imgTest2 )
print( compare )

####################################
# EMBEDDING TEST

### Using DeepFace.Verify doesn't work ###

# img1 = r"C:\Users\kanie\face-reid\VideoCaptureTask\img1.jpg"
# img2 = r"C:\Users\kanie\face-reid\VideoCaptureTask\img2.jpg"

# embedded1 = DeepFace.represent(img_path = img1)
# embedded2 = DeepFace.represent(img_path = img2)

# # compare = DeepFace.verify( embedded1, embedded2 ) --> doesn't work

### Using Cosine Distance works, and yields same results ###

# def cosine_distance(vec1, vec2):
#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)
#     return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# img1 = r"C:\Users\kanie\face-reid\VideoCaptureTask\img1.jpg"
# img2 = r"C:\Users\kanie\face-reid\VideoCaptureTask\img2.jpg"

# embedded1 = DeepFace.represent(img_path = img1)[0]["embedding"]
# embedded2 = DeepFace.represent(img_path = img2)[0]["embedding"]

# cos_dist = cosine_distance( embedded1, embedded2 )
# df_dist = DeepFace.verify( img1, img2 )["distance"]


# print( cos_dist )
# print( df_dist )

####################################

# imgTest = r"C:\Users\kanie\face-reid\VideoCaptureTask\img1.jpg"

# try:
#     faces = DeepFace.extract_faces( imgTest )
#     for i, face_dict in enumerate(faces): # loop through ever dictionary in list
#         plt.imshow(face_dict["face"]) # show the "face" of the current face_dict
#         plt.title(f"Face {i+1}")  # add title to face plot
#         plt.show()
# except:
#     print("No Faces Detected")
