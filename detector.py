from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path, conf=0.01, iou=0.45, imgsz=960, device=None, class_thresholds=None):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

        # Отдельные пороги по классам
        self.class_thresholds = {
            "Open Eye": 0.01,
            "Closed Eye": 0.01,
            "Phone": 0.10,
            "Seatbelt": 0.5
        }
        if class_thresholds:
            self.update_class_thresholds(class_thresholds)

    def update_class_thresholds(self, class_thresholds):
        for label, threshold in class_thresholds.items():
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                continue

            self.class_thresholds[label] = max(0.01, min(0.99, threshold))

    def infer(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        return results[0]

    def parse_result(self, result):
        detections = []

        names = result.names
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id)

            if isinstance(names, dict):
                label = names.get(cls_id, str(cls_id))
            else:
                label = str(cls_id)

            threshold = self.class_thresholds.get(label, self.conf)

            if float(conf) < threshold:
                continue

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "class_id": cls_id,
                "label": label,
            })

        return detections
