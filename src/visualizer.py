import streamlit as st
import cv2
import numpy as np
from landmarks import LandmarkDetector
from filters import Filter, FacemeshFilter, FACE_CONNECTIONS
import mediapipe as mp

def apply_lip_color(frame, landmarks, region_points, color, opacity=0.6):
    """
    Apply lipstick by overlaying color on the lip region using predefined region points.
    Args:
        frame (numpy.ndarray): The input image.
        landmarks: The detected facial landmarks.
        region_points (list): The list of landmark indices defining the region.
        color (tuple): The RGB color to apply (e.g., (0, 0, 255) for red).
        opacity (float): Opacity of the overlay (0.0 to 1.0).
    Returns:
        numpy.ndarray: Image with lipstick applied.
    """
    try:
        # Convert the landmark indices to pixel coordinates
        region_coords = [
            (int(landmarks.landmark[idx].x * frame.shape[1]),
             int(landmarks.landmark[idx].y * frame.shape[0]))
            for idx in region_points
        ]

        # Validate that region_coords is not empty
        if not region_coords:
            raise ValueError("Region coordinates are empty. Cannot apply makeup.")

        # Create an overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(region_coords, dtype=np.int32)], color)

        # Blend the overlay with the original frame
        return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    except Exception as e:
        raise RuntimeError(f"Error in apply_lip_color: {e}")
def apply_makeup(frame, landmarks, regions, region_name, color, opacity=0.6):
    """
    Apply makeup to a specific region defined by region_name.
    Args:
        frame (numpy.ndarray): The input image.
        landmarks: The detected facial landmarks.
        regions (dict): Dictionary containing regions and their landmark indices.
        region_name (str): The name of the region to apply makeup.
        color (tuple): The RGB color to apply.
        opacity (float): Opacity of the overlay (0.0 to 1.0).
    Returns:
        numpy.ndarray: Image with makeup applied.
    """
    if region_name not in regions:
        raise ValueError(f"Region '{region_name}' not found in regions dictionary.")

    region_points = regions[region_name]
    return apply_lip_color(frame, landmarks, region_points, color, opacity)

REGIONS = {
    "BLUSH_LEFT": [50],
    "BLUSH_RIGHT": [280],
    "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33],
    "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
    "EYELINER_LEFT": [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243],
    "EYELINER_RIGHT": [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341, 463],
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463],
    "FACE": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 454, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152],
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    "EYEBROW_LEFT": [55, 107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
    "EYEBROW_RIGHT": [285, 336, 296, 334, 293, 300, 276, 283, 295, 285]
}


def main():
    st.title("AR Makeup Application")

    # Add input options
    input_option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])
    uploaded_image = None
    if input_option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
    elif input_option == "Use Camera":
        camera_image = st.camera_input("Take a photo using your camera")
        if camera_image:
            uploaded_image = camera_image
    frame = None  # Initialize the frame variable
    # Add a dropdown menu for filter selection
    filter_option = st.selectbox(
        "Choose a filter to apply",
        ["Facemesh", "Lipstick"]
    )

    # Add opacity control for Lipstick filter
    if filter_option == "Lipstick":
        opacity = st.slider("Adjust Filter Opacity (for Lipstick)", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

    if uploaded_image:
        # Load the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if filter_option == "Facemesh":
            # Apply the facemesh filter
            facemesh_filter = FacemeshFilter()
            image_with_facemesh = facemesh_filter.detect_and_draw_facemesh(image)
            st.image(image_with_facemesh, channels="BGR", caption="Facemesh Filter Applied")


        elif filter_option == "Lipstick":
            # Apply the lipstick filter
            detector = LandmarkDetector()
            landmarks = detector.detect_landmarks(image)

        if landmarks:
            try:
                image_with_lipstick = apply_lip_color(image, landmarks, (0, 0, 255), opacity=opacity)
                st.image(image_with_lipstick, channels="BGR", caption="Lipstick Filter Applied")
            except Exception as e:
                st.error(f"An error occurred while applying the lipstick filter: {e}")
            try:
                frame = apply_makeup(frame, landmarks, REGIONS, region_name, rgb_color)
                st.image(frame, channels="BGR", caption=f"Applied makeup to {region_name}")
            except Exception as e:
                st.error(f"Error applying makeup: {e}")
        else:
            st.warning("No landmarks detected. Please upload a clear image.")
        # Dropdown for region selection
        region_name = st.selectbox("Choose a region to apply makeup", list(REGIONS.keys()))

        # Color picker for makeup
        color = st.color_picker("Pick a color for the makeup", "#FF0000")
        rgb_color = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    


if __name__ == "__main__":
    main()
