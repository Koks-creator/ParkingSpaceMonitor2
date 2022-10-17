from dataclasses import dataclass
from typing import List, Tuple
import time
import numpy as np
import cv2

np.random.seed(20)


@dataclass
class ObjectDetectorSSDMobileNet:
    config_path: str
    model_path: str
    classes_path: str
    conf_threshold: float = 0.3
    nms_threshold: float = 0.2
    img_width: int = 416
    img_height: int = 416

    def __post_init__(self):
        self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize(self.img_width, self.img_height)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.classes_list, self.color_list = self.load_classes()

    def load_classes(self) -> [List, List]:
        with open(self.classes_path) as f:
            classes_list = f.read().splitlines()

        classes_list.insert(0, '__Background__')
        color_list = np.random.randint(0, 255, size=(len(classes_list), 3))

        return classes_list, color_list

    def detect(self, img: np.array, allowed_classes=None, draw=False) -> List[Tuple]:
        if allowed_classes is None:
            allowed_classes = [i for i in range(len(self.classes_list))]

        detections_list = []

        class_labels_ids, confidences, bboxs = self.net.detect(img, confThreshold=self.conf_threshold)

        bboxs = list(bboxs)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))

        bbox_ids = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=.5, nms_threshold=self.nms_threshold)

        if len(bbox_ids):
            for i in range(0, len(bbox_ids)):
                x, y, w, h = bboxs[np.squeeze(bbox_ids[i])]
                class_conf = confidences[np.squeeze(bbox_ids[i])]
                class_label_id = np.squeeze(class_labels_ids[np.squeeze(bbox_ids[i])])
                class_label = self.classes_list[class_label_id]
                class_color = [int(c) for c in self.color_list[class_label_id]]

                detections_list.append((x, y, w, h, class_label, class_conf))

                if draw:
                    if class_label_id in allowed_classes:
                        if class_conf > self.conf_threshold:
                            display_text = f"{class_label}: {int(round(class_conf, 2) * 100)}%"

                            cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)

                            line_w = min(int(w * 0.3), int(h * 0.3))

                            cv2.line(img, (x, y), (x + line_w, y), class_color, 5)
                            cv2.line(img, (x, y), (x, y + line_w), class_color, 5)

                            cv2.line(img, (x + w, y), (x + w - line_w, y), class_color, 5)
                            cv2.line(img, (x + w, y), (x + w, y + line_w), class_color, 5)

                            cv2.line(img, (x, y + h), (x + line_w, y + h), class_color, 5)
                            cv2.line(img, (x, y + h), (x, y + h - line_w), class_color, 5)

                            cv2.line(img, (x + w, y + h), (x + w - line_w, y + h), class_color, 5)
                            cv2.line(img, (x + w, y + h), (x + w, y + h - line_w), class_color, 5)
        return detections_list


if __name__ == '__main__':
    od = ObjectDetectorSSDMobileNet(
        config_path=r"ModelSSD\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt",
        model_path=r"ModelSSD\frozen_inference_graph.pb",
        classes_path=r"dnn_model\coco.names"

    )
    img = cv2.imread("parking.jpg")
    od.detect(img, draw=True, allowed_classes=[3])

    cv2.imshow("res", img)
    cv2.waitKey(0)