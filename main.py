from time import time
from typing import List, Tuple
import cv2
import numpy as np
import pickle
from carpark2.object_detector import ObjectDetector
from carpark2.object_detectorSSD import ObjectDetectorSSDMobileNet

with open("CarParkPosKrzywe2", "rb") as f:
    pos_list = pickle.load(f)

parking_spaces = {}
for index, pos in enumerate(pos_list):
    parking_spaces[index] = {"pos": pos, "color": None, "Free": None}


def draw_parking_view_prettified(empty_img: np.array, space_w: int, space_h: int, start_x: int, start_y: int):

    off_x = 0
    off_y = 0
    per_row = 640 // (start_x + space_w)

    for index, p_space in enumerate(parking_spaces.items()):
        p_space_id, p_data = p_space
        pos, color, free = p_data.values()

        x1, y1 = start_x + off_x, start_y + off_y
        x2, y2 = x1 + space_w, y1 + space_h
        cx, cy = int(x1 + (space_w // 2)), int(y1 + (space_h // 2))

        cv2.rectangle(empty_img, (x1, y1), (x2, y2), color, -1)
        cv2.putText(empty_img, f"{p_space_id}", (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        off_x += start_x + space_w
        if index == per_row - 1:
            off_y += space_h + 25
            off_x = 0


def draw_parking_view(empty_img: np.array):
    for p_space in parking_spaces.items():
        p_space_id, p_data = p_space
        pos, color, free = p_data.values()

        m = cv2.moments(pos)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        cv2.polylines(empty_img, [pos], True, color, 2)
        cv2.fillPoly(empty_img, [pos], color)
        cv2.putText(empty_img, f"{p_space_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)


def check_parking_space(img: np.array, parking_id: int, center_points: List[Tuple[int, int]], thickness=4):
    for center_point in center_points:
        h, w, _ = img.shape
        pos, color, _ = parking_spaces[parking_id].values()

        result = cv2.pointPolygonTest(pos, center_point, False)

        if result >= 0:
            parking_spaces[parking_id]["color"] = (0, 0, 200)
            parking_spaces[parking_id]["Free"] = False
            cv2.polylines(img, [pos], True, (0, 0, 200), thickness)
            break
        else:
            parking_spaces[parking_id]["color"] = (0, 200, 0)
            parking_spaces[parking_id]["Free"] = True
            cv2.polylines(img, [pos], True, (0, 200, 0), thickness)


cap = cv2.VideoCapture(r"giga_parking.wmv")
# od = ObjectDetector(
#     weights_path=r"dnn_model\yolov4.weights",
#     cfg_path=r"dnn_model\yolov4.cfg",
#     coco_file_path=r"dnn_model\coco.names",
#     nms_threshold=.3,
#     conf_threshold=.1,
#     image_h=320,
#     image_w=320
# )

od = ObjectDetectorSSDMobileNet(
    config_path=r"ModelSSD\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt",
    model_path=r"ModelSSD\frozen_inference_graph.pb",
    classes_path=r"dnn_model\coco.names",
    nms_threshold=.28,
    conf_threshold=.1,
    img_width=512,
    img_height=512,
)


ptime = 0
parking_spaces_num = len(pos_list)
while True:
    success, img = cap.read()
    if success is False:
        break

    h, w, _ = img.shape
    empty_reg_view = np.zeros((h, w, 3), np.uint8)
    empty_pret_view = np.zeros((480, 640, 3), np.uint8)

    detections = od.detect(img, draw=True, allowed_classes=[3, 8])
    if detections:
        center_points = []

        for detection in detections:
            x, y, w, h = detection[:4]

            cx = int(x + (w//2))
            cy = int(y + (h//2))

            cy = cy + 30

            center_points.append((cx, cy))
            cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)

        for parking_id in parking_spaces.keys():
            check_parking_space(img, parking_id, center_points)

    draw_parking_view(empty_reg_view)
    draw_parking_view_prettified(empty_pret_view, space_w=100, space_h=150, start_x=50, start_y=50)

    ctime = time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime

    taken_parks = [data["Free"] for data in parking_spaces.values()].count(False)
    cv2.putText(img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Free: {taken_parks}/{parking_spaces_num}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(empty_reg_view, f"Free: {taken_parks}/{parking_spaces_num}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(empty_pret_view, f"Free: {taken_parks}/{parking_spaces_num}", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Res", img)
    cv2.imshow("ParkingSpacesView", empty_reg_view)
    cv2.imshow("PrettfiedParkingSpacesView", empty_pret_view)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
