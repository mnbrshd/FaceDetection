import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionConf=0.5):
        self.min_detection_confidence = minDetectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence)

    def findFaces(self, img, draw=True):
        self.results = self.faceDetection.process(img)

        bboxs = []
        if self.results.detections:
            for id, det in enumerate(self.results.detections):
                bboxC = det.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                bboxs.append([id, bbox, det.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(det.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        return bboxs, img
    
    def fancyDraw(self, img, bbox, length=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        #TOP LEFT
        cv2.line(img, (x,y), (x+length, y), (255, 0, 255), thickness=t)
        cv2.line(img, (x,y), (x, y+length), (255, 0, 255), thickness=t)

        #TOP RIGHT
        cv2.line(img, (x1,y), (x1-length, y), (255, 0, 255), thickness=t)
        cv2.line(img, (x1,y), (x1, y+length), (255, 0, 255), thickness=t)

        #BOTTOM LEFT
        cv2.line(img, (x,y1), (x+length, y1), (255, 0, 255), thickness=t)
        cv2.line(img, (x,y1), (x, y1-length), (255, 0, 255), thickness=t)

        #BOTTOM RIGHT
        cv2.line(img, (x1,y1), (x1-length, y1), (255, 0, 255), thickness=t)
        cv2.line(img, (x1,y1), (x1, y1-length), (255, 0, 255), thickness=t)

        return img
    
