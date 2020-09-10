import face_recognition
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os
import shutil

new_model = tf.keras.models.load_model('mask_detector.model')
Known_Faces_Dir = "known_faces"

Tolerance = 0.6
Frame_Thickness = 3
Font_Thickness = 2
Model = "hog" #cnn
Check_If_Match_Exists = []
Match_Color = [[0, 255, 0],
                [255, 0, 0],
                [0, 0, 255],
                [125, 125, 0],
                [125, 0, 125],
                [0, 125, 125]]

def predicting(image, match):
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32)/255.0
    image = np.expand_dims(image, axis=0)
    prediction_result = np.argmax(new_model.predict(image))
    #print(f"{prediction_result} -- {match} -- YOU SHOULD READ ME -- 1 = No Mask and 0 = Mask")
    if prediction_result == 1:
        return 'No Mask'
    return 'Mask'

cap = cv2.VideoCapture("testMask.gif")
#cap = cv2.VideoCapture("testNoMask.gif")
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

print("Loading known faces")

known_faces = []
known_names = []

if len(known_names) > 0:
    next_id = max(known_names) + 1
else:
    next_id = 0

ret, image = cap.read()

print("This may take a few seconds")

while True:

    ret, image = cap.read()

    if not ret:
        print("Check")
        break

    else:
        locations = face_recognition.face_locations(image, model=Model)
        encodings = face_recognition.face_encodings(image, locations)
        
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, Tolerance)
            match = None
            if True in results:
                match = known_names[results.index(True)]
                
            else:
                match = str(next_id)
                next_id+= 1
                known_names.append(match)
                known_faces.append(face_encoding)
                os.mkdir(f"{Known_Faces_Dir}/{match}")

                crop_image = image[face_location[0]: face_location[2], face_location[3]: face_location[1]].copy()
                result = predicting(crop_image, match)

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), Frame_Thickness)

            cv2.rectangle(image, top_left, bottom_right, [0, 255, 0], 2)
            cv2.putText(image, result, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), Font_Thickness)

        cv2.imshow("", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

shutil.rmtree(Known_Faces_Dir)
os.mkdir(Known_Faces_Dir)
cap.release()
cv2.destroyAllWindows()