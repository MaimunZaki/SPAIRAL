import cv2
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model=YOLO("best.pt")

results= model.predict(source="VideoTest.mp4", show=True)

cv2.imshow("best", results)

cv2.destroyAllWindows()