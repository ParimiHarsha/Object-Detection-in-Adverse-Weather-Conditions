import cv2
import numpy as np
from ultralytics import YOLO

best_model = YOLO(
    "/Users/parimi/Documents/TAMU/RA_Work/runs/detect/train2/weights/best.pt"
)
cap = cv2.VideoCapture("/Users/parimi/Documents/TAMU/RA_Work/snow_driving.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = best_model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    for bbox in bboxes:
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
