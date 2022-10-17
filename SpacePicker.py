import cv2
import pickle
import numpy as np

# Wymiary jednego miejsca parkingowe na obrazku
# width, height = 107, 48

parking_places_file = "CarParkPosKrzywe2"
try:
    with open(parking_places_file, 'rb') as f:
        pos_list = pickle.load(f)
except Exception:
    pos_list = []


points = []


def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        for index, pos in enumerate(pos_list):
            if type(pos) == tuple:
                x1, y1 = pos[:2]
                w_rec, h_rec = pos[2:]
                if x1 < x < x1 + w_rec and y1 < y < y1 + h_rec:
                    pos_list.pop(index)
            else:
                pos_region = np.array([pos[0], pos[1], pos[2], pos[3]], np.int32)

                min_x = np.min(pos_region[:, 0])
                max_x = np.max(pos_region[:, 0])
                min_y = np.min(pos_region[:, 1])
                max_y = np.max(pos_region[:, 1])

                if min_x < x < max_x and min_y < y < max_y:
                    pos_list.pop(index)


def nothing(x):
    pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("4points_space", "Tracking", 0, 1, nothing)
cv2.createTrackbar("draw", "Tracking", 1, 1, nothing)

while True:
    draw_from_4p = cv2.getTrackbarPos("4points_space", "Tracking")
    draw = cv2.getTrackbarPos("draw", "Tracking")

    img = cv2.imread("parking.jpg")
    img = cv2.resize(img, (1280, 720))
    # print(img.shape)

    for pos in pos_list:
        if draw == 1:
            if type(pos) == tuple:
                w_, h_ = pos[2:]
                cv2.rectangle(img, pos[:2], (pos[0] + w_, pos[1] + h_), (255, 0, 255), 2)
            else:
                cv2.polylines(img, [pos], True, 255, 4)

    if len(points) == 4:
        pos_region = np.array([points[0], points[1], points[2], points[3]], np.int32)

        pos_list.append(pos_region)
        points = []
    else:
        for point in points:
            cv2.circle(img, point, 8, (255, 0, 200), -1)

    with open(parking_places_file, 'wb') as f:
        pickle.dump(pos_list, f)

    cv2.imshow("res", img)
    cv2.setMouseCallback("res", mouse_click)
    cv2.waitKey(1)
