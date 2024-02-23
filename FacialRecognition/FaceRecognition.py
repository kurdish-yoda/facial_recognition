import cv2 
import dlib
import numpy as np
import json
import base64
import face_recognition as faceRegLib
import sys

#read database 
JSONpath=sys.argv[1]
readJSON = open(JSONpath)
encoded_data = json.load(readJSON)

#decode base64 string data
for documents in encoded_data['documents']:
    decoded_data_01=base64.b64decode((documents["chip_photo"]))
    
decoded_data = base64.b64decode((encoded_data["facial_image"]))

#write the decoded data back to original format in  file
img_file = open('facial_img.jpeg', 'wb')
img_file.write(decoded_data)
img_file.close()

img_file01 = open('chip_photo.jpeg', 'wb')
img_file01.write(decoded_data_01)
img_file01.close()

#Read images, encode them
cv2ReadImg = cv2.imread('facial_img.jpeg')
img_file = cv2.cvtColor(cv2ReadImg, cv2.COLOR_BGR2RGB)
img_face = faceRegLib.face_locations(img_file)[0]
facial_img_enc = faceRegLib.face_encodings(img_file)[0]

cv2ReadImg2 = cv2.imread('chip_photo.jpeg')
img_file01 = cv2.cvtColor(cv2ReadImg2, cv2.COLOR_BGR2RGB)
chip_photo_enc = faceRegLib.face_encodings(img_file01)[0]

#compare encodings and give answer
if faceRegLib.compare_faces([chip_photo_enc],facial_img_enc) == [True]:
    print("MATCH FOUND:")
    for documents in encoded_data['documents']:
        print(documents["first_name"] + " " + documents["last_name"])
else:
    print("MATCH NOT FOUND")

cv2.imshow('pic', img_file)
cv2.imshow('pic2', img_file01)

cv2.waitKey(0)

cv2.destroyAllWindows()