import cv2
import mediapipe as mp
import time
from FaceDetectionModule import FaceDetector

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceDetector()
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxs, img = detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 110), cv2.FONT_HERSHEY_PLAIN, 10, (0,255,0), 5)
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Image', img)
        cv2.waitKey(1)