import streamlit as st
import cv2
import numpy as np
from landmarks import LandmarkDetector
from filters import Filter

def main():
    st.title("AR Makeup Application")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_image:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Initialize
        detector = LandmarkDetector()
        landmarks = detector.detect_landmarks(image)

        if landmarks:
            points = detector.get_points(landmarks, image.shape)

            # Apply a filter
            filter_instance = Filter("filters/red_lipstick.png", "filters/lips_annotations.csv")
            image_with_filter = filter_instance.overlay_filter(image, points)

            st.image(image_with_filter, channels="BGR")

if __name__ == "__main__":
    main()