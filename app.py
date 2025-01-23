import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("AR Makeup Application with MediaPipe")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # For iris and lips detection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Select Makeup Style
makeup_styles = ["None", "Red Lips", "Blue Eyeshadow", "Highlight Cheeks"]
selected_style = st.selectbox("Choose a Makeup Style", makeup_styles)

# Camera Input
image_input = st.camera_input("Take a photo or upload one")

if image_input:
    # Read the image
    file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Process the frame
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh for debugging
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            # Apply makeup styles
            if selected_style == "Red Lips":
                frame = apply_lip_color(frame, face_landmarks, (255, 0, 0))  # Red
            elif selected_style == "Blue Eyeshadow":
                frame = apply_eye_shadow(frame, face_landmarks, (0, 0, 255))  # Blue
            elif selected_style == "Highlight Cheeks":
                frame = apply_cheek_highlight(frame, face_landmarks, (255, 255, 0))  # Yellow

    # Display the processed frame
    st.image(frame, channels="RGB")

# Helper Functions
def apply_lip_color(frame, landmarks, color):
    """Apply lipstick by overlaying color on the lip region."""
    lips = mp_face_mesh.FACEMESH_LIPS
    lip_points = [
        (int(landmarks.landmark[idx].x * frame.shape[1]),
         int(landmarks.landmark[idx].y * frame.shape[0]))
        for idx in lips
    ]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(lip_points, dtype=np.int32)], color)
    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

def apply_eye_shadow(frame, landmarks, color):
    """Apply eyeshadow by overlaying color on the eye regions."""
    eyes = mp_face_mesh.FACEMESH_LEFT_EYE.union(mp_face_mesh.FACEMESH_RIGHT_EYE)
    eye_points = [
        (int(landmarks.landmark[idx].x * frame.shape[1]),
         int(landmarks.landmark[idx].y * frame.shape[0]))
        for idx in eyes
    ]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(eye_points, dtype=np.int32)], color)
    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

def apply_cheek_highlight(frame, landmarks, color):
    """Apply highlight on the cheeks by overlaying color on cheek regions."""
    cheeks = [mp_face_mesh.FACEMESH_LEFT_CHEEK, mp_face_mesh.FACEMESH_RIGHT_CHEEK]
    for cheek in cheeks:
        cheek_points = [
            (int(landmarks.landmark[idx].x * frame.shape[1]),
             int(landmarks.landmark[idx].y * frame.shape[0]))
            for idx in cheek
        ]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(cheek_points, dtype=np.int32)], color)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    return frame
