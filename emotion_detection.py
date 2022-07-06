import numpy as np
import cv2
from keras.preprocessing import image
import os
import tensorflow as tf


#opencv initialization
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
#---------------------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open('facial_expression_structure.json',"r").read())
model.load_weights("model_weights.h5")

#-----------------------------------
emotions = ('angry','disgust','fear','happy','sad','surprise','neutral')
result = [0]*7
result1 = [0.0]*7

while True:
    succes,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if(len(faces)==0):
        cv2.putText(img, "No Face Detected", (10,(1*20 +8)),font,0.5,(255,0,255),2)
    else:
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#draw rectangle to main image
            detected_face = img[int(y):int(y+h),int(x):int(x+w)]#crop detected face
            detected_face = cv2.cvtColor(detected_face,cv2.COLOR_BGR2GRAY)#TRANSFORM INTO GRAYSCALE
            detected_face = cv2.resize(detected_face,(48,48))
            
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels,axis = 0)
            #img_pixels = 255
            predictions = model.predict(img_pixels) #store probilities of 7 expressions
            max_index = np.argmax(predictions[0])#find max indexed array

            emotion = emotions[max_index]
            result[max_index] = result[max_index]+1
            result1[max_index] = result1[max_index] + round(predictions[0][max_index]*100,2)

            y1 = y
            y = y + 20
            #write emotions text above rectangle
            cv2.putText(img,emotion,(int(x),int(y1)),font,1,(255,0,255),2)
            cv2.putText(img,"Emotions",(10,10),font,0.5,(0,0,0),2)
            cv2.putText(img,"| Probility",(100,10),font,0.5,(0,0,0),2)
            cv2.putText(img,"| Probility Bar",(200,10),font,0.5,(0,0,0),2)

            bar = ".............................................................."
            z = 0
            for i in range(len(predictions[0])):
                b = bar[0:int(predictions[0][i]*len(bar))]
                if(i == max_index):
                    cv2.putText(img,emotions[i] + ": ",(10,(i+1)*20+8),font,0.5,(255,0,255),2)
                    cv2.putText(img,"| ",(100,(i+1)*20+8),font,0.5,(0,0,0),2)
                    cv2.putText(img," "+ str(round(predictions[0][i]*100,2)),(110,(i+1)*20+8),font,0.5,(255,255,255),2)
                    cv2.putText(img,"| ",(200,(i+1)*20+8),font,0.5,(0,0,0),2)

                    cv2.rectangle(img,(210, i*20 + 10),(210 + int(predictions[0][i]*100),(i+1)*20 +4),(255,0,0),-1)

                else:
                    cv2.putText(img,emotions[i] + ": ",(10,(i+1)*20+8),font,0.5,(0,255,0),2)
                    cv2.putText(img,"| ",(100,(i+1)*20+8),font,0.5,(0,0,0),2)
                    cv2.putText(img," "+ str(round(predictions[0][i]*100,2)),(110,(i+1)*20+8),font,0.5,(255,0,0),2)
                    cv2.putText(img,"| ",(200,(i+1)*20+8),font,0.5,(0,0,0),2)

                    cv2.rectangle(img,(210, i*20 + 10),(210 + int(predictions[0][i]*100),(i+1)*20 +4),(255,0,0),-1)
                print("Detected Mood is:",emotions[i])
    cv2.imshow('img',img)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


