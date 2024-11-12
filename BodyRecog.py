import cv2
import numpy as np
from moviepy.editor import *
import tensorflow as tf

from tensorflow import keras

from keras.layers import *
from keras.models import load_model

#HOG 
#HOG (Histogram of Oriented Gradients) is an object detector used to detect objects in computer vision and image processing. The technique counts the occurrence of gradient orientation in localised portions of an image. The cv.HOGDescriptor() method creates the HOG descriptor. The hog.setSVMDetector() method sets coefficients for the linear SVM classifier.

#path to video
path = r"E:\Projects\Baby\Media\Dataset 2.0\Walking\Walking (2).mp4"

#opening the video
cap = cv2.VideoCapture(path)

#import pre-trained model
saved_model = load_model('LSTM_Model.h5')

#declare image height and width
img_ht, img_width = 64, 64

#actions
classes = ["Walk", "Run"]

#used for the prediction of action
def predict_action(video_file_path):
    #read the video
    video_reader = cv2.VideoCapture(video_file_path)

    frames_list = []

    predicted_class_name = ''

    #for each in a 20 seq of frames
    for frame_counter in range(20):
        success, frame = video_reader.read()

        if not success:
            break 

        resized_frame = cv2.resize(frame, (img_ht, img_width))

        normalized_frame = resized_frame/255

        frames_list.append(normalized_frame)

    # Reshape frames for LSTM input (samples, time_steps, features)
    reshaped_frames = np.array(frames_list).reshape(1, 20, img_ht * img_width * 3) #1, 20, imght* img wd* 3

    predicted_labels_probabilities = saved_model.predict(reshaped_frames)[0]
    predict_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = classes[predict_label]

    # \nConfidence: {predicted_labels_prrobabilities[predict_label]}
    print(f'Action Predicted: {predicted_class_name}')

    video_reader.release()
    

def dec_intruder(humans):
    if(humans == 1):
        print("SUSPICIOUS PERSON DETECTED!!!")
        predict_action(path)

hog = cv2.HOGDescriptor()
#Support vector Machine used for people detection
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)
    
    # cap.write(frame.astype('uint8'))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

    (humans, _) = hog.detectMultiScale(frame,
                                   winStride=(5, 5),
                                   padding=(3, 4),
                                   scale=1.21)

    dec_intruder(len(humans))

    print('Human detected: ', len(humans))

cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
