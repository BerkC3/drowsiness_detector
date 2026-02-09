import os

import cv2
import numpy as np
import dlib
import torch
from imutils import face_utils
from scipy.spatial import distance as dist

from model import EyeStateModel


class FaceProcessor:
    """Dlib face/landmark detection + CNN eye-state inference."""

    LEFT_EYE_START, LEFT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    RIGHT_EYE_START, RIGHT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    MOUTH_START, MOUTH_END = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    INNER_MOUTH_START, INNER_MOUTH_END = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    _MIN_ROI_DIM = 8

    def __init__(self, predictor_path: str, model_path: str, img_size: tuple = (24, 24)) -> None:
        if not os.path.isfile(predictor_path):
            raise FileNotFoundError(
                f"Dlib shape predictor not found: {predictor_path}\n"
                "Download from http://dlib.net/files/"
                "shape_predictor_68_face_landmarks.dat.bz2"
            )

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(predictor_path)
        self._img_size = img_size
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"CNN model not found: {model_path}\n"
                "Run train_model.py first to generate drowsiness_model.pth"
            )
        try:
            self._device = torch.device("cpu")
            self._model = EyeStateModel()
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load CNN model: {exc}") from exc

    @staticmethod
    def compute_ear(eye: np.ndarray) -> float:
        """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
        vert_a = dist.euclidean(eye[1], eye[5])
        vert_b = dist.euclidean(eye[2], eye[4])
        horiz = dist.euclidean(eye[0], eye[3])
        if horiz < 1e-6:
            return 0.0
        return (vert_a + vert_b) / (2.0 * horiz)

    @staticmethod
    def compute_mar(inner_mouth: np.ndarray) -> float:
        """MAR = (||p1-p7|| + ||p2-p6|| + ||p3-p5||) / (2 * ||p0-p4||)"""
        vert_a = dist.euclidean(inner_mouth[1], inner_mouth[7])
        vert_b = dist.euclidean(inner_mouth[2], inner_mouth[6])
        vert_c = dist.euclidean(inner_mouth[3], inner_mouth[5])
        horiz = dist.euclidean(inner_mouth[0], inner_mouth[4])
        if horiz < 1e-6:
            return 0.0
        return (vert_a + vert_b + vert_c) / (2.0 * horiz)

    def preprocess_eye(self, eye_roi: np.ndarray) -> torch.Tensor:
        if len(eye_roi.shape) == 3:
            eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        eye_roi = cv2.resize(eye_roi, self._img_size)
        eye_roi = self._clahe.apply(eye_roi)
        eye_roi = eye_roi.astype("float32") / 255.0
        return torch.from_numpy(eye_roi).unsqueeze(0).unsqueeze(0)

    def predict_eye_state(self, eye_roi: np.ndarray) -> float:
        """Sigmoid confidence: ~0 = closed, ~1 = open. Returns -1.0 on failure."""
        if eye_roi is None or eye_roi.size == 0:
            return -1.0
        if min(eye_roi.shape[:2]) < self._MIN_ROI_DIM:
            return -1.0
        try:
            tensor = self.preprocess_eye(eye_roi).to(self._device)
            with torch.no_grad():
                logits = self._model(tensor)
                prob = torch.sigmoid(logits)
            return float(prob.item())
        except Exception:
            return -1.0

    def detect_faces(self, gray_frame: np.ndarray):
        return self._detector(gray_frame, 0)

    def predict_landmarks(self, gray_frame: np.ndarray, face_rect) -> np.ndarray:
        shape = self._predictor(gray_frame, face_rect)
        return face_utils.shape_to_np(shape)

    def get_eye_landmarks(self, landmarks: np.ndarray):
        left = landmarks[self.LEFT_EYE_START : self.LEFT_EYE_END]
        right = landmarks[self.RIGHT_EYE_START : self.RIGHT_EYE_END]
        return left, right

    def get_mouth_landmarks(self, landmarks: np.ndarray):
        mouth = landmarks[self.MOUTH_START : self.MOUTH_END]
        inner = landmarks[self.INNER_MOUTH_START : self.INNER_MOUTH_END]
        return mouth, inner
