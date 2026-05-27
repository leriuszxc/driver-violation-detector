import cv2
import threading
import time
from queue import Full


class CameraWorker(threading.Thread):
    """
    Поток чтения одной камеры.
    Кладет свежие кадры в очередь.
    Если очередь переполнена, новые кадры пропускаются,
    чтобы не накапливалась большая задержка.
    """

    def __init__(self, camera_id, source, frame_queue, stop_event):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.cap = None

    def open_camera(self):
        # Если источник - число (веб-камера 0, 1, 2), используем DSHOW (для Windows)
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        # Если источник - строка (IP-камера rtsp:// или http://), DSHOW не используем
        else:
            cap = cv2.VideoCapture(self.source)
            
        if not cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть камеру {self.camera_id} (source={self.source})"
            )
        return cap
    
    def run(self):
        while not self.stop_event.is_set():
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = self.open_camera()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
                
                h, w = frame.shape[:2]
                scale = 1280.0 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

                item = {
                    "camera_id": self.camera_id,
                    "timestamp": time.time(),
                    "frame": frame,
                }

                try:
                    self.frame_queue.put_nowait(item)
                except Full:
                    pass

            except Exception as e:
                print(f"[CameraWorker {self.camera_id}] Ошибка: {e}")
                time.sleep(1.0)

        if self.cap is not None:
            self.cap.release()