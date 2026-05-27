import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

import cv2


class EventLogger:
    def __init__(self, csv_path, snapshot_dir, json_path=None):
        self.csv_path = csv_path
        self.snapshot_dir = snapshot_dir
        self.json_path = json_path or os.path.join(os.path.dirname(csv_path), "violations.json")
        self._lock = RLock()

        os.makedirs(self.snapshot_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["datetime", "camera_id", "event", "snapshot_path"])

        if not os.path.exists(self.json_path):
            self._write_events_unlocked(self._migrate_csv_events())

    def _migrate_csv_events(self):
        events = []
        if not os.path.exists(self.csv_path):
            return events

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt_str = row.get("datetime", "")
                camera_id = row.get("camera_id", "")
                event_name = row.get("event", "")
                snapshot_path = row.get("snapshot_path", "")
                raw_id = f"{dt_str}|{camera_id}|{event_name}|{snapshot_path}"

                events.append({
                    "id": hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16],
                    "datetime": dt_str,
                    "camera_id": self._parse_camera_id(camera_id),
                    "event": event_name,
                    "snapshot_path": snapshot_path,
                    "snapshot_file": os.path.basename(snapshot_path),
                })

        return events

    def _parse_camera_id(self, camera_id):
        try:
            return int(camera_id)
        except (TypeError, ValueError):
            return camera_id

    def _read_events_unlocked(self):
        if not os.path.exists(self.json_path):
            return []

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        if isinstance(data, list):
            return data
        return []

    def _write_events_unlocked(self, events):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        tmp_path = f"{self.json_path}.tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, self.json_path)

    def _rewrite_csv_unlocked(self, events):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["datetime", "camera_id", "event", "snapshot_path"])
            for event in events:
                writer.writerow([
                    event.get("datetime", ""),
                    event.get("camera_id", ""),
                    event.get("event", ""),
                    event.get("snapshot_path", ""),
                ])

    def _remove_snapshot_unlocked(self, snapshot_path):
        if not snapshot_path:
            return

        try:
            snapshot = Path(snapshot_path).resolve()
            snapshot_dir = Path(self.snapshot_dir).resolve()
            if snapshot_dir not in snapshot.parents:
                return
            if snapshot.exists() and snapshot.is_file():
                snapshot.unlink()
        except OSError:
            return

    def log_event(self, camera_id, event_name, frame):
        with self._lock:
            dt = datetime.now()
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            dt_file = dt.strftime("%Y%m%d_%H%M%S")
            event_id = uuid4().hex

            filename = f"{dt_file}_cam{camera_id}_{event_name}_{event_id[:8]}.jpg"
            snapshot_path = os.path.join(self.snapshot_dir, filename)

            cv2.imwrite(snapshot_path, frame)

            record = {
                "id": event_id,
                "datetime": dt_str,
                "timestamp": dt.timestamp(),
                "camera_id": camera_id,
                "event": event_name,
                "snapshot_path": snapshot_path,
                "snapshot_file": filename,
            }

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([dt_str, camera_id, event_name, snapshot_path])

            events = self._read_events_unlocked()
            events.append(record)
            self._write_events_unlocked(events)

            print(f"[EVENT] {dt_str} | camera={camera_id} | event={event_name} | saved={snapshot_path}")
            return record

    def get_events(self):
        with self._lock:
            return list(reversed(self._read_events_unlocked()))

    def get_event(self, event_id):
        with self._lock:
            for event in self._read_events_unlocked():
                if event.get("id") == event_id:
                    return event
        return None

    def delete_event(self, event_id):
        with self._lock:
            events = self._read_events_unlocked()
            kept_events = []
            deleted = None

            for event in events:
                if event.get("id") == event_id:
                    deleted = event
                else:
                    kept_events.append(event)

            if deleted is None:
                return False

            self._remove_snapshot_unlocked(deleted.get("snapshot_path"))
            self._write_events_unlocked(kept_events)
            self._rewrite_csv_unlocked(kept_events)
            return True

    def clear_events(self):
        with self._lock:
            events = self._read_events_unlocked()
            for event in events:
                self._remove_snapshot_unlocked(event.get("snapshot_path"))

            self._write_events_unlocked([])
            self._rewrite_csv_unlocked([])
            return len(events)
