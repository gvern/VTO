import mediapipe as mp
import cv2
import numpy as np

class LandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_landmarks(self, frame):
        """Detects facial landmarks in the given frame."""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def get_points(self, landmarks, image_shape):
        """Extracts pixel coordinates from landmarks."""
        h, w = image_shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]