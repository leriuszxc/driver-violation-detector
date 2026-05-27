import base64
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import cv2
import torch
import webview

from camera_worker import CameraWorker
from config import (
    CAMERA_SOURCES,
    CONF_THRESHOLD,
    DEFAULT_DETECTION_SENSITIVITY,
    DEFAULT_ENABLED_VIOLATIONS,
    DRAW_BACKEND_BOXES,
    DROWSY_SECONDS_THRESHOLD,
    FRAME_QUEUE_SIZE,
    IMG_SIZE,
    IOU_THRESHOLD,
    LOG_DIR,
    MODEL_PATH,
    NO_SEATBELT_SECONDS_THRESHOLD,
    PHONE_SECONDS_THRESHOLD,
    SAVE_VIOLATIONS,
    SEATBELT_ON_SECONDS_THRESHOLD,
    SHOW_WINDOWS,
    SNAPSHOT_DIR,
    SETTINGS_JSON_PATH,
    VIOLATIONS_JSON_PATH,
    VIRTUAL_CAMERA_IDS,
)
from detector import YOLODetector
from logger_utils import EventLogger
from rules import ViolationRules


EVENT_DISPLAY_SECONDS = 3.0
INTERFACE_PATH = Path(__file__).resolve().parent / "interface.html"

_api_lock = threading.Lock()
_api_data = {}
_runtime_lock = threading.Lock()
_runtime_detector = None
_runtime_rules = None
_runtime_logger = None
_settings_lock = threading.Lock()
_settings = None
stop_event = None


def camera_key(camera_id):
    return f"cam{camera_id}"


def clamp_sensitivity(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 50
    return max(1, min(100, value))


def sensitivity_to_threshold(sensitivity):
    return max(0.01, min(0.99, 1.0 - (clamp_sensitivity(sensitivity) / 100.0)))


def sensitivity_to_thresholds(class_sensitivity):
    return {
        label: sensitivity_to_threshold(value)
        for label, value in class_sensitivity.items()
    }


def normalize_settings(raw_settings=None):
    raw_settings = raw_settings if isinstance(raw_settings, dict) else {}
    raw_sensitivity = raw_settings.get("class_sensitivity", {})
    raw_violations = raw_settings.get("enabled_violations", {})

    class_sensitivity = {
        label: clamp_sensitivity(raw_sensitivity.get(label, value))
        for label, value in DEFAULT_DETECTION_SENSITIVITY.items()
    }
    enabled_violations = {
        key: bool(raw_violations.get(key, value))
        for key, value in DEFAULT_ENABLED_VIOLATIONS.items()
    }

    return {
        "class_sensitivity": class_sensitivity,
        "enabled_violations": enabled_violations,
    }


def load_settings():
    if not SETTINGS_JSON_PATH.exists():
        return normalize_settings()

    try:
        with open(SETTINGS_JSON_PATH, "r", encoding="utf-8") as f:
            return normalize_settings(json.load(f))
    except (json.JSONDecodeError, OSError):
        return normalize_settings()


def save_settings(settings):
    SETTINGS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = SETTINGS_JSON_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    tmp_path.replace(SETTINGS_JSON_PATH)


def get_settings_snapshot():
    with _settings_lock:
        return normalize_settings(_settings)


def update_runtime_settings():
    settings = get_settings_snapshot()
    thresholds = sensitivity_to_thresholds(settings["class_sensitivity"])

    with _runtime_lock:
        if _runtime_detector is not None:
            _runtime_detector.update_class_thresholds(thresholds)
        if _runtime_rules is not None:
            _runtime_rules.update_enabled_violations(settings["enabled_violations"])


def encode_file_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError:
        return None


def get_event_logger():
    with _runtime_lock:
        logger = _runtime_logger

    if logger is not None:
        return logger

    return EventLogger(
        csv_path=str(LOG_DIR / "events.csv"),
        snapshot_dir=str(SNAPSHOT_DIR),
        json_path=str(VIOLATIONS_JSON_PATH),
    )


class Api:

    def get_latest_data(self):
        with _api_lock:
            return dict(_api_data)

    def get_settings(self):
        return get_settings_snapshot()

    def update_class_sensitivity(self, class_name, sensitivity):
        global _settings

        with _settings_lock:
            _settings = normalize_settings(_settings)
            if class_name not in _settings["class_sensitivity"]:
                return {"ok": False, "settings": _settings}

            _settings["class_sensitivity"][class_name] = clamp_sensitivity(sensitivity)
            save_settings(_settings)

        update_runtime_settings()
        return {"ok": True, "settings": get_settings_snapshot()}

    def update_violation_enabled(self, violation_key, enabled):
        global _settings

        with _settings_lock:
            _settings = normalize_settings(_settings)
            if violation_key not in _settings["enabled_violations"]:
                return {"ok": False, "settings": _settings}

            _settings["enabled_violations"][violation_key] = bool(enabled)
            save_settings(_settings)

        update_runtime_settings()
        return {"ok": True, "settings": get_settings_snapshot()}

    def get_violation_history(self):
        logger = get_event_logger()
        events = logger.get_events()
        for event in events:
            event["snapshot_available"] = bool(event.get("snapshot_path") and Path(event["snapshot_path"]).exists())
        return events

    def get_violation_snapshot(self, event_id):
        logger = get_event_logger()
        event = logger.get_event(event_id)
        if not event:
            return {"ok": False, "frame": None}

        frame = encode_file_to_base64(event.get("snapshot_path"))
        return {"ok": frame is not None, "frame": frame, "event": event}

    def delete_violation(self, event_id):
        logger = get_event_logger()
        ok = logger.delete_event(event_id)
        return {"ok": ok, "events": self.get_violation_history()}

    def clear_violations(self):
        logger = get_event_logger()
        deleted = logger.clear_events()
        return {"ok": True, "deleted": deleted, "events": []}

    def get_violation_log_path(self):
        return str(VIOLATIONS_JSON_PATH)


def get_color(label):
    if label == "Closed Eye":
        return (0, 0, 255)
    if label == "Open Eye":
        return (0, 255, 0)
    if label == "Phone":
        return (0, 165, 255)
    if label == "Seatbelt":
        return (255, 255, 0)
    return (255, 255, 255)


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]

        color = get_color(label)
        text = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    return frame


def put_text_alpha(frame, text, org, font, scale, color, thickness, alpha):
    overlay = frame.copy()
    cv2.putText(overlay, text, org, font, scale, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_labels_list(frame, detections):
    labels = [f"{d['label']} {d['conf']:.2f}" for d in detections]

    if not labels:
        labels = ["No detections"]

    y = 30
    for text in labels[:10]:
        put_text_alpha(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            0.45,
        )
        y += 28

    return frame


def draw_events(frame, events):
    if not events:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    texts = [f"VIOLATION: {event_name}" for event_name in events]
    max_width = 0

    for text in texts:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        max_width = max(max_width, text_size[0])

    box_x1 = w - max_width - 40
    box_y1 = 10
    box_x2 = w - 10
    box_y2 = 20 + len(texts) * 35

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    y = 35
    for text in texts:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_width = text_size[0]
        x = w - text_width - 20

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 35

    return frame


def draw_datetime(frame):
    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        dt_str,
        (w - 280, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return frame


def apply_violation_tint(frame, events):
    if not events:
        return frame

    overlay = frame.copy()
    overlay[:] = (0, 0, 255)

    if len(events) == 1:
        alpha = 0.18
    elif len(events) == 2:
        alpha = 0.28
    else:
        alpha = 0.38

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def encode_frame_to_base64(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def build_ui_detections(detections, frame_width, frame_height):
    ui_detections = []
    if frame_width <= 0 or frame_height <= 0:
        return ui_detections

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(frame_width - 1, int(x1)))
        y1 = max(0, min(frame_height - 1, int(y1)))
        x2 = max(x1 + 1, min(frame_width, int(x2)))
        y2 = max(y1 + 1, min(frame_height, int(y2)))

        color = "cyan" if det["label"] in ["Seatbelt", "Open Eye"] else "red"
        ui_detections.append({
            "label": det["label"],
            "conf": round(float(det["conf"]), 3),
            "color": color,
            "x": (x1 / frame_width) * 100,
            "y": (y1 / frame_height) * 100,
            "w": ((x2 - x1) / frame_width) * 100,
            "h": ((y2 - y1) / frame_height) * 100,
        })

    return ui_detections


def get_display_camera_ids(camera_id):
    if len(CAMERA_SOURCES) == 1 and VIRTUAL_CAMERA_IDS:
        return VIRTUAL_CAMERA_IDS
    return (camera_id,)


def push_camera_data(camera_id, frame, detections, new_events):
    source_cam_id = camera_key(camera_id)
    h_frame, w_frame = frame.shape[:2]
    frame_base64 = encode_frame_to_base64(frame)

    if frame_base64 is None:
        return

    ui_detections = build_ui_detections(detections, w_frame, h_frame)
    frame_size = {"width": w_frame, "height": h_frame}
    camera_payloads = {}

    for display_camera_id in get_display_camera_ids(camera_id):
        display_cam_id = camera_key(display_camera_id)
        camera_payloads[display_cam_id] = {
            "cam_id": display_cam_id,
            "source_cam_id": source_cam_id,
            "frame": frame_base64,
            "frame_size": frame_size,
            "detections": ui_detections,
            "new_violation": new_events[0] if display_camera_id == camera_id and new_events else None,
        }

    latest_data = {
        "cam_id": source_cam_id,
        "source_cam_id": source_cam_id,
        "frame": frame_base64,
        "frame_size": frame_size,
        "detections": ui_detections,
        "new_violation": new_events[0] if new_events else None,
        "updated_at": time.time(),
        "cameras": camera_payloads,
    }

    with _api_lock:
        existing_cameras = _api_data.get("cameras", {}).copy()
        existing_cameras.update(camera_payloads)
        latest_data["cameras"] = existing_cameras
        _api_data.update(latest_data)


def inference_loop(stop_event_obj):
    global _runtime_detector, _runtime_logger, _runtime_rules

    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    event_display_until = {}
    settings = get_settings_snapshot()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    print(f"Фиксация нарушений: {'включена' if SAVE_VIOLATIONS else 'выключена'}")

    detector = YOLODetector(
        model_path=MODEL_PATH,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=device,
        class_thresholds=sensitivity_to_thresholds(settings["class_sensitivity"]),
    )

    rules = ViolationRules(
        phone_seconds_threshold=PHONE_SECONDS_THRESHOLD,
        drowsy_seconds_threshold=DROWSY_SECONDS_THRESHOLD,
        no_seatbelt_seconds_threshold=NO_SEATBELT_SECONDS_THRESHOLD,
        seatbelt_on_seconds_threshold=SEATBELT_ON_SECONDS_THRESHOLD,
        enabled_violations=settings["enabled_violations"],
    )

    logger = EventLogger(
        csv_path=str(LOG_DIR / "events.csv"),
        snapshot_dir=str(SNAPSHOT_DIR),
        json_path=str(VIOLATIONS_JSON_PATH),
    )

    with _runtime_lock:
        _runtime_detector = detector
        _runtime_rules = rules
        _runtime_logger = logger

    workers = []
    for camera_id, source in CAMERA_SOURCES.items():
        worker = CameraWorker(camera_id, source, frame_queue, stop_event_obj)
        worker.start()
        workers.append(worker)

    print("Система запущена. Интерфейс работает через pywebview.")

    try:
        while not stop_event_obj.is_set():
            try:
                item = frame_queue.get(timeout=1.0)
            except Empty:
                continue

            now = time.time()
            camera_id = item["camera_id"]
            frame = item["frame"]

            if camera_id == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            result = detector.infer(frame)
            detections = detector.parse_result(result)
            new_events = rules.evaluate(camera_id, detections)

            for event_name in new_events:
                event_display_until[(camera_id, event_name)] = now + EVENT_DISPLAY_SECONDS

            visible_events = [
                event_name
                for (event_camera_id, event_name), until_ts in event_display_until.items()
                if event_camera_id == camera_id and until_ts > now
            ]

            event_display_until = {
                key: until_ts
                for key, until_ts in event_display_until.items()
                if until_ts > now
            }

            vis_frame = frame.copy()
            vis_frame = apply_violation_tint(vis_frame, visible_events)
            if DRAW_BACKEND_BOXES:
                vis_frame = draw_detections(vis_frame, detections)
            vis_frame = draw_labels_list(vis_frame, detections)
            vis_frame = draw_events(vis_frame, visible_events)
            vis_frame = draw_datetime(vis_frame)

            if len(get_display_camera_ids(camera_id)) == 1:
                cv2.putText(
                    vis_frame,
                    f"Camera: {camera_id}",
                    (20, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            if SAVE_VIOLATIONS:
                log_frame = vis_frame
                if not DRAW_BACKEND_BOXES:
                    log_frame = draw_detections(vis_frame.copy(), detections)

                for event_name in new_events:
                    logger.log_event(camera_id, event_name, log_frame)

            push_camera_data(camera_id, vis_frame, detections, new_events)

            if SHOW_WINDOWS:
                cv2.imshow(f"Camera {camera_id}", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event_obj.set()
                    break

    except Exception as e:
        print(f"Ошибка в потоке распознавания: {e}")

    finally:
        stop_event_obj.set()
        with _runtime_lock:
            if _runtime_detector is detector:
                _runtime_detector = None
            if _runtime_rules is rules:
                _runtime_rules = None
            if _runtime_logger is logger:
                _runtime_logger = None
        for worker in workers:
            worker.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("Поток нейросети остановлен.")


def on_closed():
    print("Окно интерфейса закрыто. Завершаем работу...")
    if stop_event is not None:
        stop_event.set()


def main():
    global stop_event, _settings

    if not INTERFACE_PATH.exists():
        raise FileNotFoundError(f"Не найден файл интерфейса: {INTERFACE_PATH}")

    with _settings_lock:
        _settings = load_settings()
        save_settings(_settings)

    stop_event = threading.Event()
    yolo_thread = threading.Thread(target=inference_loop, args=(stop_event,), daemon=True)
    yolo_thread.start()

    window = webview.create_window(
        title="Система мониторинга водителя",
        url=INTERFACE_PATH.as_uri(),
        js_api=Api(),
        width=1280,
        height=720,
        resizable=True,
    )
    window.events.closed += on_closed

    webview.start()

    stop_event.set()
    yolo_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
