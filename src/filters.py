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

# Define FACE_CONNECTIONS
FACE_CONNECTIONS = frozenset([
    # Lips
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
    (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (78, 95),
    (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318),
    (318, 324), (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
    # Left eyebrow
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
    # Right eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155),
    (155, 133), (33, 246), (246, 161), (161, 160), (160, 159), (159, 158),
    (158, 157), (157, 173), (173, 133),
    # Right eyebrow
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21),
    (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
])


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

    def detect_and_draw_facemesh(self, frame):
        """Detects facial landmarks and draws the facemesh with FACE_CONNECTIONS."""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return frame  # Return the original frame if no landmarks are detected

        for face_landmarks in results.multi_face_landmarks:
            for connection in FACE_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = face_landmarks.landmark[start_idx]
                end_point = face_landmarks.landmark[end_idx]
                start_coord = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
                end_coord = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
                cv2.line(frame, start_coord, end_coord, (0, 255, 0), 1)  # Draw the connection lines

            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  # Draw the landmark points

        return frame