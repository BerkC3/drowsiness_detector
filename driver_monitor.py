import os
import time
import threading

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

from config import Config
from face_processor import FaceProcessor


class DriverMonitor:

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or Config()
        self._face_processor = FaceProcessor(
            self._cfg.PREDICTOR_PATH,
            self._cfg.MODEL_PATH,
            self._cfg.IMG_SIZE,
        )
        self._cap: cv2.VideoCapture | None = None
        self._frame_counter: int = 0
        self._yawn_counter: int = 0
        self._head_drop_counter: int = 0
        self._failed_reads: int = 0
        self._last_beep_time: float = 0.0

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def start(self, source=None) -> None:
        if source is None:
            source = self._cfg.CAMERA_INDEX

        if isinstance(source, str):
            ext = os.path.splitext(source)[1].lower()
            if ext in self._IMAGE_EXTENSIONS:
                self._run_on_image(source)
                return

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open source: {source}")
            return

        print("[INFO] Drowsiness detection started. Press 'q' to quit.")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    self._failed_reads += 1
                    if self._failed_reads >= self._cfg.MAX_FAILED_FRAMES:
                        print("[ERROR] Camera lost: too many consecutive read failures.")
                        break
                    continue

                self._failed_reads = 0
                processed = self._process_frame(frame)
                cv2.imshow(self._cfg.WINDOW_NAME, processed)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._cleanup()

    def _run_on_image(self, path: str) -> None:
        frame = cv2.imread(path)
        if frame is None:
            print(f"[ERROR] Cannot read image: {path}")
            return

        print("[INFO] Image detection running. Press any key to close.")
        processed = self._process_frame(frame)
        cv2.imshow(self._cfg.WINDOW_NAME, processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_processor.detect_faces(gray)

        if len(faces) == 0:
            self._frame_counter = 0
            self._yawn_counter = 0
            self._head_drop_counter = 0
            self._draw_no_face_warning(frame)
            return frame

        primary = max(faces, key=lambda r: r.width() * r.height())
        landmarks = self._face_processor.predict_landmarks(gray, primary)
        left_eye, right_eye = self._face_processor.get_eye_landmarks(landmarks)

        # eye-state via CNN
        left_roi = self._extract_eye_roi(gray, left_eye)
        right_roi = self._extract_eye_roi(gray, right_eye)
        left_conf = self._face_processor.predict_eye_state(left_roi)
        right_conf = self._face_processor.predict_eye_state(right_roi)

        valid = [c for c in (left_conf, right_conf) if c >= 0.0]
        if len(valid) == 2:
            avg_conf = (left_conf + right_conf) / 2.0
            both_closed = (
                left_conf < self._cfg.EYE_CONFIDENCE_THRESHOLD
                and right_conf < self._cfg.EYE_CONFIDENCE_THRESHOLD
            )
        elif len(valid) == 1:
            avg_conf = valid[0]
            both_closed = False
        else:
            avg_conf = -1.0
            both_closed = False

        is_drowsy = self._update_drowsiness_state(both_closed)

        # yawn detection
        mouth, inner_mouth = self._face_processor.get_mouth_landmarks(landmarks)
        mar = FaceProcessor.compute_mar(inner_mouth)
        is_yawning = self._update_yawn_state(mar)

        # head pose
        pitch, yaw, roll, nose_line = self._face_processor.estimate_head_pose(
            landmarks, frame.shape,
        )
        is_head_drop = self._update_head_drop_state(pitch)

        # sound alerts (priority: drowsy > head drop > yawn)
        if is_drowsy:
            self._play_alert_sound(*self._cfg.DROWSY_BEEP)
        elif is_head_drop:
            self._play_alert_sound(*self._cfg.HEAD_DROP_BEEP)
        elif is_yawning:
            self._play_alert_sound(*self._cfg.YAWN_BEEP)

        self._draw_overlays(
            frame, avg_conf, left_eye, right_eye, is_drowsy,
            mar, mouth, is_yawning,
            pitch, nose_line, is_head_drop,
        )
        return frame

    @staticmethod
    def _extract_eye_roi(gray: np.ndarray, eye_landmarks: np.ndarray) -> np.ndarray | None:
        """Square crop around the eye, sized by eye width (more stable than height)."""
        (x, y, w, h) = cv2.boundingRect(eye_landmarks)
        cx, cy = x + w // 2, y + h // 2

        side = int(w * 1.8)  # include eyebrow + under-eye context
        half = side // 2

        fh, fw = gray.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(fw, cx + half)
        y2 = min(fh, cy + half)

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # zero-pad back to square when the eye is near the frame edge
        rh, rw = roi.shape[:2]
        if rh != rw:
            target = max(rh, rw)
            padded = np.zeros((target, target), dtype=roi.dtype)
            pad_y = (target - rh) // 2
            pad_x = (target - rw) // 2
            padded[pad_y:pad_y + rh, pad_x:pad_x + rw] = roi
            roi = padded

        return roi

    def _update_drowsiness_state(self, both_closed: bool) -> bool:
        if both_closed:
            self._frame_counter += 1
        else:
            self._frame_counter = 0
        return self._frame_counter >= self._cfg.CONSEC_FRAMES

    def _update_yawn_state(self, mar: float) -> bool:
        if mar > self._cfg.MAR_THRESHOLD:
            self._yawn_counter += 1
        else:
            self._yawn_counter = 0
        return self._yawn_counter >= self._cfg.YAWN_CONSEC_FRAMES

    def _update_head_drop_state(self, pitch: float) -> bool:
        if pitch < self._cfg.HEAD_PITCH_THRESHOLD:
            self._head_drop_counter += 1
        else:
            self._head_drop_counter = 0
        return self._head_drop_counter >= self._cfg.HEAD_DROP_CONSEC_FRAMES

    def _play_alert_sound(self, freq: int, duration: int) -> None:
        if winsound is None or not self._cfg.SOUND_ENABLED:
            return
        now = time.time()
        if now - self._last_beep_time < self._cfg.ALERT_SOUND_COOLDOWN:
            return
        self._last_beep_time = now
        threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

    # -- drawing helpers --

    def _draw_overlays(
        self, frame, eye_conf, left_eye, right_eye, is_drowsy,
        mar=0.0, mouth=None, is_yawning=False,
        pitch=0.0, nose_line=None, is_head_drop=False,
    ) -> None:
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, self._cfg.EYE_CONTOUR_COLOR, 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, self._cfg.EYE_CONTOUR_COLOR, 1)

        if mouth is not None:
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, self._cfg.MOUTH_CONTOUR_COLOR, 1)

        # nose direction line
        if nose_line is not None:
            cv2.line(frame, nose_line[0], nose_line[1], self._cfg.HEAD_POSE_LINE_COLOR, 2)

        if eye_conf < 0:
            eye_text = "Eye: N/A"
        else:
            state = "Open" if eye_conf >= self._cfg.EYE_CONFIDENCE_THRESHOLD else "Closed"
            eye_text = f"Eye: {state} ({eye_conf:.2f})"

        yawn_state = "Yawning" if mar > self._cfg.MAR_THRESHOLD else "Normal"
        mouth_text = f"Mouth: {yawn_state} ({mar:.2f})"
        head_text = f"Head Angle: {pitch:.1f} deg"

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 8), (310, 105), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, eye_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, self._cfg.FONT_SCALE_EAR,
                    self._cfg.EAR_TEXT_COLOR, self._cfg.FONT_THICKNESS)
        cv2.putText(frame, mouth_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, self._cfg.FONT_SCALE_EAR,
                    self._cfg.MAR_TEXT_COLOR, self._cfg.FONT_THICKNESS)
        cv2.putText(frame, head_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, self._cfg.FONT_SCALE_EAR,
                    self._cfg.EAR_TEXT_COLOR, self._cfg.FONT_THICKNESS)

        h, w = frame.shape[:2]
        if is_head_drop:
            self._put_centered_text(frame, "HEAD DROP ALERT!", w, h - 110,
                                    self._cfg.HEAD_DROP_ALERT_COLOR)
        if is_yawning:
            self._put_centered_text(frame, "YAWNING ALERT!", w, h - 70,
                                    self._cfg.YAWN_ALERT_TEXT_COLOR)
        if is_drowsy:
            self._put_centered_text(frame, "DROWSINESS ALERT!", w, h - 30,
                                    self._cfg.ALERT_TEXT_COLOR)

    def _put_centered_text(self, frame, text, frame_w, y, color):
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  self._cfg.FONT_SCALE_ALERT, self._cfg.FONT_THICKNESS)
        x = (frame_w - size[0]) // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    self._cfg.FONT_SCALE_ALERT, color, self._cfg.FONT_THICKNESS)

    def _draw_no_face_warning(self, frame):
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, self._cfg.FONT_SCALE_EAR,
                    self._cfg.ALERT_TEXT_COLOR, self._cfg.FONT_THICKNESS)

    def _cleanup(self):
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Drowsiness detection stopped.")
