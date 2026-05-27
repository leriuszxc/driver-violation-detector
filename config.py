from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lastPROD2.pt"

LOG_DIR = BASE_DIR / "logs"
SNAPSHOT_DIR = LOG_DIR / "snapshots"
RECORD_DIR = BASE_DIR / "output" / "recordings"
VIOLATIONS_JSON_PATH = LOG_DIR / "violations.json"
SETTINGS_JSON_PATH = LOG_DIR / "settings.json"

LOG_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
RECORD_DIR.mkdir(parents=True, exist_ok=True)

# ========================================================
# ВЫБОР КАМЕР
# ========================================================
CAMERA_SOURCES = {
    0: 0,
}

# Test mode: read one physical camera and show it as several virtual
# streams without extra capture/model passes.
VIRTUAL_CAMERA_IDS = (0, 1, 2)

# The HTML UI draws boxes over the video. Keeping backend boxes off avoids
# duplicate rectangles in JPEG and DOM.
DRAW_BACKEND_BOXES = False

DEFAULT_DETECTION_SENSITIVITY = {
    "Open Eye": 99,
    "Closed Eye": 99,
    "Phone": 90,
    "Seatbelt": 50,
}

DEFAULT_ENABLED_VIOLATIONS = {
    "seatbelt": True,
    "phone": True,
    "drowsiness": True,
}
# ВАРИАНТ 2: Использование 3-х IP камер
# Обычно это RTSP (rtsp://логин:пароль@IP:порт/поток) или HTTP/MJPEG

# CAMERA_SOURCES = {
#
# }

# Параметры модели
IMG_SIZE = 640
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

# Пороги по времени в секундах
PHONE_SECONDS_THRESHOLD = 2.0
DROWSY_SECONDS_THRESHOLD = 1.5
NO_SEATBELT_SECONDS_THRESHOLD = 3.0
SEATBELT_ON_SECONDS_THRESHOLD = 1.0

# Очередь кадров
FRAME_QUEUE_SIZE = 3

# Отображение окна
SHOW_WINDOWS = False

# Запись видео, если потом понадобится
SAVE_RECORDINGS = False

# Главный переключатель фиксации нарушений:
# True  -> сохранять изображения и писать события в CSV
# False -> только показывать нарушения на экране, без сохранения
SAVE_VIOLATIONS = True
