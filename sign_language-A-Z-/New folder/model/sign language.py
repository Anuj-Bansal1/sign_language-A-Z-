import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow import keras

model = keras.models.load_model(".\sign_model.h5")

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands

label = False

hands = mpHands.Hands(max_num_hands=1)

mpdraw = mp.solutions.drawing_utils
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
while True:
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
        roi = cv.cvtColor(roi,cv.COLOR_RGB2GRAY)
        roi = (np.expand_dims(roi,0))
        
       # print(roi.shape)
        
        prediction = model.predict(roi) 
        label = np.argmax(prediction)
    if label :
        cv.putText(frame,classes[label],(localx,localy-10),cv.FONT_HERSHEY_COMPLEX,3,(0,255,0),4,cv.LINE_AA)        
    cv.imshow("final" , frame)
    if cv.waitKey(25) == 27 :
        break

cap.release()
cv.destroyAllWindows()