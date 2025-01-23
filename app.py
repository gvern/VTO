import streamlit as st
import torch
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

st.title("GAN-Based Makeup Application")
st.write("Apply realistic makeup styles using GANs and facial landmark detection.")

# Load ProgressiveGAN model
@st.cache_resource
def load_model():
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-256')
    return model

model = load_model()
latent_dim = getattr(model, "latent_size", 512)

# MediaPipe for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)

# Sidebar: Makeup Styles
styles = ["Natural", "Red Lips", "Blue Eyeshadow", "Smoky Eyes"]
selected_style = st.sidebar.selectbox("Choose a makeup style", styles)

# File uploader for the input image
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "png", "jpeg"])

# Helper functions for makeup application
def apply_lip_color(frame, landmarks, color):
    lips_idx = mp_face_mesh.FACEMESH_LIPS
    lip_points = [(int(landmarks.landmark[i].x * frame.shape[1]), 
                   int(landmarks.landmark[i].y * frame.shape[0])) 
                  for i in lips_idx]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(lip_points, dtype=np.int32)], color)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    return frame

def apply_eye_shadow(frame, landmarks, color):
    eye_idx = mp_face_mesh.FACEMESH_RIGHT_EYE + mp_face_mesh.FACEMESH_LEFT_EYE
    eye_points = [(int(landmarks.landmark[i].x * frame.shape[1]), 
                   int(landmarks.landmark[i].y * frame.shape[0])) 
                  for i in eye_idx]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(eye_points, dtype=np.int32)], color)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    return frame

# Process uploaded image
if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    # Convert image to numpy array for processing
    image_np = np.array(input_image)
    results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Apply selected makeup styles
            if selected_style == "Red Lips":
                image_np = apply_lip_color(image_np, face_landmarks, (0, 0, 255))  # Red
            elif selected_style == "Blue Eyeshadow":
                image_np = apply_eye_shadow(image_np, face_landmarks, (255, 0, 0))  # Blue
            elif selected_style == "Smoky Eyes":
                image_np = apply_eye_shadow(image_np, face_landmarks, (50, 50, 50))  # Gray

    st.image(image_np, caption=f"Styled Image: {selected_style}", use_container_width=True)

# Footer
st.write("Powered by ProgressiveGAN, MediaPipe, and Streamlit")
