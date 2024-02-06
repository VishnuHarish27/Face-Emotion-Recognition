import cv2 as cv
from facial_emotion_recognition import EmotionRecognition 
emotion_recognition =  EmotionRecognition(device = 'cpu')
cam = cv.VideoCapture(0)
while True:
    _, frame = cam.read()
    frame = emotion_recognition.recognise_emotion(frame, return_type = 'BGR')
    cv.imshow ('emotion recognition', frame)
    key = cv.waitKey(1)
    if key == 27:
        break
cam.release()
cv.destroyAllWindows()