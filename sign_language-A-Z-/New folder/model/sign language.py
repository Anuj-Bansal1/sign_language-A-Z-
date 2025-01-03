import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow import keras
# Use a pipeline as a high-level helper
from transformers import pipeline

from PIL import Image

pipe = pipeline("image-classification", model="dima806/hand_gestures_image_detection")
# pipe = pipeline("image-classification", model="Hemg/sign-language-classification")
model = keras.models.load_model(".\sign_model.h5")

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands

label = False

hands = mpHands.Hands(max_num_hands=1)

mpdraw = mp.solutions.drawing_utils
# classes = ["Unity","Busy","Cute","Dance","help","Nice","You","Direction","Promise","Judge","victory","Love","fist","fist bump","Ok","Rabbit","Pick","Promise","Unity","Unity","Up","Victory","Three","You","Yolo","That"]

frame_c = 0
while True:
    frame_c += 1
    ret,frame = cap.read()
    frame = cv.flip(frame,1)
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = hands.process(img)
    
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(frame,handlms,mpHands.HAND_CONNECTIONS) 
            for id,lm in enumerate( handlms.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x *w),int(lm.y *h)
                if id == 8:
                    localx,localy = (cx-45),(cy-60)

                if id == 20:
                    finalx,finaly = (cx+50),(cy+200)
            #img = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
            roi =  frame[(localy):(finaly),(localx):(finalx)] 
        cv.imshow("roi",roi)
        cv.rectangle(frame,(localx,localy),(finalx,finaly),(255,45,213),3)
        #print(roi.shape)
        roi = cv.resize(roi,(28,28))
        roi2 = frame
        roi2 = cv.cvtColor(roi2, cv.COLOR_BGR2RGB)
        roi2 = Image.fromarray(roi2)
        roi = cv.cvtColor(roi,cv.COLOR_RGB2GRAY)

        roi = (np.expand_dims(roi,0))
        
       # print(roi.shape)
        
        prediction = model.predict(roi) 
        if frame_c%2 == 0:
            # label = np.argmax(prediction)
            label = pipe(roi2)
    if label :
        # cv.putText(frame,classes[label],(localx,localy-10),cv.FONT_HERSHEY_COMPLEX,3,(0,255,0),4,cv.LINE_AA)        
        cv.putText(frame,label[0]["label"],(localx,localy-10),cv.FONT_HERSHEY_COMPLEX,3,(0,255,0),4,cv.LINE_AA)    
        print(label)    
    cv.imshow("final" , frame)
    if cv.waitKey(25) == 27 :
        break

cap.release()
cv.destroyAllWindows()