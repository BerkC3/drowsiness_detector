import os


class Config:
    # EAR (kept for fallback/reference)
    EAR_THRESHOLD: float = 0.25
    CONSEC_FRAMES: int = 48

    # MAR
    MAR_THRESHOLD: float = 0.45
    YAWN_CONSEC_FRAMES: int = 20

    # dlib shape predictor
    PREDICTOR_PATH: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "shape_predictor_68_face_landmarks.dat",
    )

    # CNN eye-state model
    MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "drowsiness_model.pth",
    )
    IMG_SIZE: tuple = (24, 24)
    EYE_CONFIDENCE_THRESHOLD: float = 0.5

    # Head pose
    HEAD_PITCH_THRESHOLD: float = -15.0  # degrees, negative = head dropping forward
    HEAD_DROP_CONSEC_FRAMES: int = 30
    HEAD_POSE_LINE_COLOR: tuple = (255, 0, 0)
    HEAD_DROP_ALERT_COLOR: tuple = (0, 0, 255)

    # Sound alerts (frequency_hz, duration_ms)
    SOUND_ENABLED: bool = True
    ALERT_SOUND_COOLDOWN: float = 1.5  # seconds between beeps
    DROWSY_BEEP: tuple = (1000, 400)
    YAWN_BEEP: tuple = (600, 300)
    HEAD_DROP_BEEP: tuple = (1200, 400)

    CAMERA_INDEX: int = 0
    MAX_FAILED_FRAMES: int = 30

    # BGR colours
    EYE_CONTOUR_COLOR: tuple = (0, 255, 0)
    MOUTH_CONTOUR_COLOR: tuple = (255, 0, 255)
    ALERT_TEXT_COLOR: tuple = (0, 0, 255)
    YAWN_ALERT_TEXT_COLOR: tuple = (0, 165, 255)
    EAR_TEXT_COLOR: tuple = (255, 255, 255)
    MAR_TEXT_COLOR: tuple = (255, 255, 255)

    FONT_SCALE_EAR: float = 0.7
    FONT_SCALE_ALERT: float = 1.0
    FONT_THICKNESS: int = 2

    WINDOW_NAME: str = "Driver Drowsiness Monitor"
