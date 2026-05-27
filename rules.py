from collections import defaultdict
import time


class ViolationRules:
    def __init__(
        self,
        phone_seconds_threshold=2.0,
        drowsy_seconds_threshold=1.5,
        no_seatbelt_seconds_threshold=3.0,
        seatbelt_on_seconds_threshold=1.0,
        enabled_violations=None,
    ):
        self.phone_seconds_threshold = phone_seconds_threshold
        self.drowsy_seconds_threshold = drowsy_seconds_threshold
        self.no_seatbelt_seconds_threshold = no_seatbelt_seconds_threshold
        self.seatbelt_on_seconds_threshold = seatbelt_on_seconds_threshold
        self.enabled_violations = {
            "seatbelt": True,
            "phone": True,
            "drowsiness": True,
        }
        if enabled_violations:
            self.update_enabled_violations(enabled_violations)

        self.state = defaultdict(lambda: {
            "phone_since": None,
            "drowsy_since": None,

            "seatbelt_on_since": None,
            "seatbelt_off_since": None,

            # unknown / belt_on / belt_off
            "seatbelt_state": "unknown",

            # Чтобы стартовое отсутствие ремня фиксировалось только один раз
            "initial_no_seatbelt_reported": False,

            "last_events": set(),
        })

    def update_enabled_violations(self, enabled_violations):
        for key in self.enabled_violations:
            if key in enabled_violations:
                self.enabled_violations[key] = bool(enabled_violations[key])

    def evaluate(self, camera_id, detections):
        now = time.time()
        labels = [d["label"] for d in detections]
        st = self.state[camera_id]
        events = []

        # ---------------------------------
        # 1. Телефон
        # ---------------------------------
        if self.enabled_violations["phone"] and "Phone" in labels:
            if st["phone_since"] is None:
                st["phone_since"] = now
        else:
            st["phone_since"] = None

        if st["phone_since"] is not None:
            if now - st["phone_since"] >= self.phone_seconds_threshold:
                events.append("phone_usage")

        # ---------------------------------
        # 2. Сонливость: только ДВА закрытых глаза
        # ---------------------------------
        closed_eye_count = sum(1 for d in detections if d["label"] == "Closed Eye")

        if self.enabled_violations["drowsiness"] and closed_eye_count >= 2:
            if st["drowsy_since"] is None:
                st["drowsy_since"] = now
        else:
            st["drowsy_since"] = None

        if st["drowsy_since"] is not None:
            if now - st["drowsy_since"] >= self.drowsy_seconds_threshold:
                events.append("drowsiness")

        # ---------------------------------
        # 3. Ремень безопасности
        # ---------------------------------
        seatbelt_detected = "Seatbelt" in labels

        if not self.enabled_violations["seatbelt"]:
            st["seatbelt_on_since"] = None
            st["seatbelt_off_since"] = None
        elif seatbelt_detected:
            if st["seatbelt_on_since"] is None:
                st["seatbelt_on_since"] = now
            st["seatbelt_off_since"] = None
        else:
            if st["seatbelt_off_since"] is None:
                st["seatbelt_off_since"] = now
            st["seatbelt_on_since"] = None

        # Подтверждаем состояние "ремень надет"
        if st["seatbelt_on_since"] is not None:
            if now - st["seatbelt_on_since"] >= self.seatbelt_on_seconds_threshold:
                st["seatbelt_state"] = "belt_on"

        # Подтверждаем состояние "ремень отсутствует"
        if st["seatbelt_off_since"] is not None:
            if now - st["seatbelt_off_since"] >= self.no_seatbelt_seconds_threshold:

                # Если в начале ремня нет 3 секунды — фиксируем стартовое нарушение
                if st["seatbelt_state"] == "unknown" and not st["initial_no_seatbelt_reported"]:
                    events.append("initial_no_seatbelt")
                    st["seatbelt_state"] = "belt_off"
                    st["initial_no_seatbelt_reported"] = True

                # Если раньше ремень был, а потом исчез — фиксируем снятие ремня
                elif st["seatbelt_state"] == "belt_on":
                    events.append("seatbelt_removed")
                    st["seatbelt_state"] = "belt_off"

        # ---------------------------------
        # 4. Защита от спама одинаковыми событиями
        # ---------------------------------
        current_events = set(events)
        new_events = []

        for event_name in current_events:
            if event_name not in st["last_events"]:
                new_events.append(event_name)

        st["last_events"] = current_events
        return new_events
