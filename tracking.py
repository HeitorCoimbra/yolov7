import cv2
import numpy as np
from detect_and_retrieve import Detector
import math

# Initialize Object Detection
detector = Detector('best.pt', 640, 0.5, 0.5)


cap = cv2.VideoCapture("test/testy.mp4")

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    count += 1
    if not ret:
        break

    center_points_cur_frame = []

    boxes = detector.detect(frame)[0][0]
    for box in boxes:
        (x1, y1, x2, y2) = [int(i) for i in box[:4]]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue
                
            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")

    cv2.imshow("Frame", frame)
    
    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
