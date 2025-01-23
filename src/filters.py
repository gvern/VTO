import cv2
import numpy as np
import mediapipe as mp
from utils import load_annotations, load_image_with_alpha

class Filter:
    def __init__(self, filter_path, annotation_path):
        self.filter_path = filter_path
        self.annotation_path = annotation_path
        self.filter_image, self.alpha_channel = load_image_with_alpha(self.filter_path)
        self.annotations = load_annotations(self.annotation_path)

    def overlay_filter(self, frame, landmarks, opacity=0.8):
        """Overlays the filter on the frame."""
        for annotation in self.annotations:
            points = [landmarks[idx] for idx in annotation]
            h, w = self.filter_image.shape[:2]

            # Define transformation matrix
            dst_points = np.array(points, dtype=np.float32)
            src_points = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)
            transform_matrix = cv2.getAffineTransform(src_points, dst_points[:3])

            # Warp the filter image
            warped_filter = cv2.warpAffine(self.filter_image, transform_matrix, (frame.shape[1], frame.shape[0]))
            warped_alpha = cv2.warpAffine(self.alpha_channel, transform_matrix, (frame.shape[1], frame.shape[0]))

            # Apply Gaussian blur to alpha for better blending
            blurred_alpha = cv2.GaussianBlur(warped_alpha, (15, 15), 10)
            alpha_mask = (blurred_alpha / 255.0) * opacity

            # Blend the filter with the frame
            for c in range(3):  # For each color channel
                frame[:, :, c] = frame[:, :, c] * (1 - alpha_mask) + warped_filter[:, :, c] * alpha_mask

        return frame


class FacemeshFilter:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Includes iris and lips detection
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def overlay_facemesh(self, frame):
        """Detects and draws a facemesh overlay on the frame."""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return frame  # Return the original frame if no landmarks are detected

        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
        return frame
