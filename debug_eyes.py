"""
Visual debugger for eye ROI extraction and CNN predictions.
Shows raw ROI, preprocessed 24x24, and model confidence side by side.
Press 'q' to quit, 's' to save a snapshot.
"""

import os
import cv2
import numpy as np
import torch
import dlib
from imutils import face_utils

from model import EyeStateModel
from config import Config


def load_training_samples(dataset_dir, count=3):
    samples = {}
    for cls in ["Closed_Eyes", "Open_Eyes"]:
        path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(path):
            continue
        files = [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        samples[cls] = []
        for f in files[:count]:
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                samples[cls].append(cv2.resize(img, (80, 80)))
    return samples


def main():
    cfg = Config()
    device = torch.device("cpu")

    model = EyeStateModel()
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(cfg.PREDICTOR_PATH)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    l_start, l_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    r_start, r_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    training_samples = load_training_samples(dataset_dir)

    strip_parts = []
    for cls, imgs in training_samples.items():
        for img in imgs:
            border_val = 255 if cls == "Open_Eyes" else 128
            strip_parts.append(cv2.copyMakeBorder(img, 2, 2, 2, 2,
                                                  cv2.BORDER_CONSTANT, value=border_val))
    if strip_parts:
        cv2.imshow("Training Samples", np.hstack(strip_parts))

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Debug mode. 'q' = quit, 's' = save snapshot.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        if len(faces) == 0:
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        face = max(faces, key=lambda r: r.width() * r.height())
        landmarks = face_utils.shape_to_np(predictor(gray, face))
        left_pts = landmarks[l_start:l_end]
        right_pts = landmarks[r_start:r_end]

        for name, pts in [("Left", left_pts), ("Right", right_pts)]:
            (x, y, w, h) = cv2.boundingRect(pts)
            cx, cy = x + w // 2, y + h // 2
            half = int(w * 1.8) // 2
            fh, fw = gray.shape[:2]

            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(fw, cx + half), min(fh, cy + half)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_big = cv2.resize(roi, (160, 80), interpolation=cv2.INTER_NEAREST)

            processed = clahe.apply(cv2.resize(roi, (24, 24)))
            tensor = torch.from_numpy(processed.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                prob = torch.sigmoid(model(tensor)).item()

            proc_big = cv2.resize(processed, (80, 80), interpolation=cv2.INTER_NEAREST)

            roi_bgr = cv2.cvtColor(roi_big, cv2.COLOR_GRAY2BGR)
            proc_bgr = cv2.cvtColor(proc_big, cv2.COLOR_GRAY2BGR)

            label = f"{'Open' if prob > 0.5 else 'Closed'} ({prob:.3f})"
            cv2.putText(roi_bgr, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(proc_bgr, "24x24", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow(f"{name} Eye ROI", np.hstack([roi_bgr, proc_bgr]))

            color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
            cv2.drawContours(frame, [cv2.convexHull(pts)], -1, color, 1)
            y_pos = 30 if name == "Left" else 60
            cv2.putText(frame, f"{name}: {prob:.3f}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(f"  {name}: roi={roi.shape}, min={roi.min()}, max={roi.max()}, "
                  f"prob={prob:.4f}", end="")

        print()

        cv2.imshow("Debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("debug_snapshot.png", frame)
            print("[SAVED] debug_snapshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
