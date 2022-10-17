from dataclasses import dataclass
from typing import List, Union
import cv2
import numpy as np


@dataclass
class ObjectDetector:
    weights_path: str
    cfg_path: str
    coco_file_path: str
    nms_threshold: float = .3
    conf_threshold: float = .3
    image_w: int = 416
    image_h: int = 416

    def __post_init__(self):
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.class_names = self.load_classes()

    def load_classes(self) -> List[str]:
        with open(self.coco_file_path) as f:
            class_names = f.read().strip("\n").split("\n")
            return class_names

    def detect(self, img: np.array, allowed_classes=False, draw=False) -> List[List[Union[int, int, int, int, str]]]:
        ih, iw, _ = img.shape

        bbox = []
        class_ids = []
        confs = []

        if allowed_classes is False:
            allowed_classes = [i for i in range(len(self.class_names))]

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.image_w, self.image_h), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_names)

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id in allowed_classes:
                    if confidence > self.conf_threshold:
                        w, h = int(det[2] * iw), int(det[3] * ih)
                        x, y = int((det[0] * iw) - w / 2), int((det[1] * ih) - h / 2)

                        bbox.append([x, y, w, h])
                        class_ids.append(class_id)
                        confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, self.conf_threshold, self.nms_threshold)

        bbox_list = []
        for i in indices:
            i = i[0]

            box = bbox[i]
            x, y, w, h = box

            class_name = self.class_names[class_ids[i]].upper()

            bbox_list.append([x, y, w, h, class_name, confs[i]])

            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (240, 100, 255), 2)
                cv2.putText(img, f"{self.class_names[class_ids[i]].upper()} {int(confs[i] * 100)}%", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return bbox_list


if __name__ == '__main__':
    od = ObjectDetector(
        weights_path=r"dnn_model\yolov4.weights",
        cfg_path=r"dnn_model\yolov4.cfg",
        coco_file_path=r"dnn_model\coco.names"
    )

    img = cv2.imread("parking.jpg")
    od.detect(img, draw=True)

    cv2.imshow("res", img)
    cv2.waitKey(0)