import streamlit as st
import cv2
import numpy as np
from landmarks import LandmarkDetector
from filters import Filter, FacemeshFilter


def main():
    st.title("AR Makeup Application")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

    # Add a dropdown menu for filter selection
    filter_option = st.selectbox(
        "Choose a filter to apply",
        ["Facemesh", "Lipstick"]
    )

    if uploaded_image:
        # Load the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if filter_option == "Facemesh":
            # Apply the facemesh filter
            facemesh_filter = FacemeshFilter()
            image_with_facemesh = facemesh_filter.overlay_facemesh(image)
            st.image(image_with_facemesh, channels="BGR", caption="Facemesh Filter Applied")

        elif filter_option == "Lipstick":
            # Apply the lipstick filter
            filter_instance = Filter("filters/red_lipstick.png", "filters/lips_annotations.csv")
            detector = LandmarkDetector()
            landmarks = detector.detect_landmarks(image)
            if landmarks:
                points = detector.get_points(landmarks, image.shape)
                image_with_filter = filter_instance.overlay_filter(image, points)
                st.image(image_with_filter, channels="BGR", caption="Lipstick Filter Applied")
            else:
                st.warning("No landmarks detected. Please upload a clear image.")


if __name__ == "__main__":
    main()
