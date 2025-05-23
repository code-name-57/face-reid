####################################
# LIBRARIES
from deepface import DeepFace
import cv2
import time
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
####################################

####################################
# FUNCTIONS

# To Compare / Verify Faces (default in DeepFace)
def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# To Compare / Verify Faces (another option, but may need different thresholds?)
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

####################################

####################################
# VARIABLE DEFINITIONS

# Changeable
videoPath = r"C:\Users\kanie\face-reid\VideoCaptureTask\HomeAloneClip.mp4"
K = 3   # For K-MEANS CLUSTERING (will add 'best k value' code below as well)

# Pre-Defined
frameNum = 1            # Total Number of Frames
faceNum = 0             # Total Number of Faces From All Frames
uniquePersonNum = 0     # Total Number of Unique Faces

embeddings = []         # Every Face Embedding
people = []             # List (people) of list (their individual embeddings)


####################################
# OPEN VIDEO
video = cv2.VideoCapture( videoPath ) 
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
                #cv2.imwrite(r"C:\Users\kanie\face-reid\VideoCaptureTask\FacesInFrames\Face" + str(faceNum) + ".jpg", frame)
                try:
                    curImgEmbedded = DeepFace.represent(
                        face_dict["face"],
                        enforce_detection=False
                    )[0]["embedding"]

                    embeddings.append({
                        "frame": frameNum,
                        "embedding": curImgEmbedded
                    })

                except Exception as e:
                    print(f"Could not get embedding: {e}")
                    continue

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

####################################
# K-MEANS CLUSTERING

# Save embeddings into own array
data = []
for j in embeddings:
    data.append( np.array( j["embedding"] ) )

print(f"Total face embeddings found: {len(data)}")

# Determine best K value
K_range = range(2, len(data) )
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_scores.append(score)

# Pick the best K
best_k = K_range[np.argmax(silhouette_scores)]
print(f"Best K (by max silhouette score): {best_k}")

if not (best_k == K):
    print("Estimated K-Value does NOT match given")
    best_k = K
    print("Adjusted K value to stay as given value")

# Create People Labels
labels = []
tally = []
for i in range( best_k ):
    label = "Person" + str(i)
    labels.append(label) 
    tally.append(0)

# PLOTTING VISUAL GRAPH OF CLUSTERS

kmeans = KMeans(n_clusters=best_k, random_state=0).fit(data)
cluster_labels = kmeans.labels_

pca = PCA(n_components=2)

# Need to Reduce to 2D for visualization
reduced = pca.fit_transform(data)
reduced_centers = pca.transform(kmeans.cluster_centers_)

# Plot/Graph details
plt.figure(figsize=(8, 5))
plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='rainbow')
plt.title(f"Visualized Clusters (K={best_k})")
for i, txt in enumerate(labels):
    plt.annotate(txt, (reduced[i, 0]+0.2, reduced[i, 1]))

# Add one label per cluster
for i, (x, y) in enumerate(reduced_centers):
    plt.annotate(f"Person{i}", (x, y), fontsize=10, weight='bold', color='black')

plt.show()

####################################
# Organising Data Further

# Add extra key in embeddings dictionary for which cluster each one belongs to
for i, entry in enumerate(embeddings):
    person_id = cluster_labels[i]
    entry["person_id"] = labels[person_id]
    # print("Face", i, "from Frame", entry["frame"], "assigned to", labels[person_id])
    tally[ person_id ] += 1

    # entry["person_id"] = labels[i]
    # print("Face ", i, "matches", i["person_id"]) --> didn't work (try again later)

print("********")
for i in range( len(tally) ):
    print( labels[person_id], "Appeared", tally[i], "Times" )

####################################

#  other stuff

# img1 = r"C:\Users\kanie\face-reid\VideoCaptureTask\img1.jpg"
# img2 = r"C:\Users\kanie\face-reid\VideoCaptureTask\img2.jpg"

# embedded1 = DeepFace.represent(img_path = img1)[0]["embedding"]
# embedded2 = DeepFace.represent(img_path = img2)[0]["embedding"]

# cos_dist = cosine_distance( embedded1, embedded2 )
# df_dist = DeepFace.verify( img1, img2 )["distance"]


# print( cos_dist )
# print( df_dist )

####################################
########     SELF-NOTES     ########
# CURRENTLY
# -> Process a video / go through a video frame-by-frame
# -> Extract faces from each frame
# -> Convert the face into an embedding
# -> Put all embeddings into an array (with other data)
# -> Use k-means clustering to identify who is who at once
#   --> can predict k but not very accurate (will be overridden if incorrect)
# -> Display (2D) graph of all face-embedding data points in clusters
# -> Tag each face to which cluster they belong to (and print it out)
#
# IN THE FUTURE
# -> Other Displays
# -> Put in a Database rather than just arrays
# -> Eliminate unneeded data points
#   --> Consider cosine similarity matrix ?
#   --> ☆☆☆ K NEAREST NEIGHBOURS ☆☆☆
# -> Make k-best value estimate more accurate? somehow?
# 
####################################
