from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

##import cascade library(downloaded from udemy course on python and machine learning)
cascade_classifier = cv2.CascadeClassifier('/home/chime/PycharmProjects/Testthesis/haarcascade_frontalface_default.xml')
## defined a model saved from the training session
model = load_model('/home/chime/PycharmProjects/Testthesis/Emotions.h5')
## defining the labels of emotions available
classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
# initilaising camera(laptop camera= 0, external=1)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/chime/PycharmProjects/Testthesis/frankietruthh.mp4')
while True:
    ## return the frame being seen by the camera
    ret, face = cap.read()
    labels = []
    ##convert bgr to gray for easy comparism with model
    bgr2gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    ##Reduce the frame of the of the image and detect features using knn
    face_rects = cascade_classifier.detectMultiScale(bgr2gray,1.3,5)

    for (x,y,w,h) in face_rects:
        ##white solid rectangle for facemapping and thickness of 10
        cv2.rectangle(face, (x, y), (x+w, y+h), (255, 255, 255), 2)
        gray_roi = bgr2gray[y:y+h, x:x+w]
        gray_roi = cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_AREA)



        if np.sum([gray_roi])!=0:
            roi = gray_roi.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

        # fetch from the classes and predict the microexpression

            prediction = model.predict(roi)[0]
            microexpression = classes[prediction.argmax()]
            label_position = (x,y)
            ##Text color and type for the microexpression
            cv2.putText(face, microexpression, label_position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 5)
        else:
            ##Text color and type if no face to detect microexpression
            cv2.putText(face, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 0), 2)
    cv2.imshow('Micro-expression', face)
    ##quit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
