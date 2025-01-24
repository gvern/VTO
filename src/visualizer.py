import streamlit as st
import cv2
import numpy as np
from landmarks import LandmarkDetector
from filters import FacemeshFilter


def apply_makeup(image, landmarks, regions, region_colors, region_opacities):
    """
    Apply a mask filter to the image based on facial landmarks with per-region opacity.
    Args:
        image (np.array): The input image.
        landmarks: The detected facial landmarks (mediapipe object).
        regions (dict): Regions with corresponding facial landmark indices.
        region_colors (dict): Colors for each region (BGR format).
        region_opacities (dict): Opacity/intensity of the makeup overlay for each region.
    Returns:
        np.array: Image with the mask filter applied.
    """
    try:
        mask = np.zeros_like(image, dtype=np.uint8)

        for region_name, region_points in regions.items():
            if region_name in region_colors:
                # Extract region coordinates
                region_coords = [
                    (
                        int(landmarks[idx].x * image.shape[1]),
                        int(landmarks[idx].y * image.shape[0])
                    )
                    for idx in region_points if idx < len(landmarks)
                ]

                if region_coords:
                    # Apply color to the mask
                    cv2.fillPoly(mask, [np.array(region_coords, dtype=np.int32)], region_colors[region_name])

        # Apply opacity per region and blend the mask
        result_image = image.copy()
        for region_name, opacity in region_opacities.items():
            if region_name in region_colors:
                mask_region = np.zeros_like(image, dtype=np.uint8)
                region_coords = [
                    (
                        int(landmarks[idx].x * image.shape[1]),
                        int(landmarks[idx].y * image.shape[0])
                    )
                    for idx in regions[region_name] if idx < len(landmarks)
                ]
                if region_coords:
                    cv2.fillPoly(mask_region, [np.array(region_coords, dtype=np.int32)], region_colors[region_name])
                    mask_region = cv2.GaussianBlur(mask_region, (7, 7), 4)
                    result_image = cv2.addWeighted(result_image, 1, mask_region, opacity, 0)

        return result_image

    except Exception as e:
        raise RuntimeError(f"Error in apply_makeup: {e}")


REGIONS = {
    "BLUSH_LEFT": [50],
    "BLUSH_RIGHT": [280],
    "EYES": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33,
              362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
    "EYEBROWS": [55, 107, 66, 105, 63, 70, 46, 53, 52, 65, 55,
                 285, 336, 296, 334, 293, 300, 276, 283, 295, 285],
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463],
    "FACE": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
             338, 297, 332, 284, 251, 389, 454, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152],
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    "BLUSH": [280, 50],  # Grouped blush
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

    if uploaded_image:
        # Load the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Original Image")  # Preview original image

        # Filter options
        filter_option = st.selectbox("Choose a filter to apply", ["Facemesh", "Custom Makeup"])

        detector = LandmarkDetector()
        landmarks = detector.detect_landmarks(image)

        if filter_option == "Facemesh":
            facemesh_filter = FacemeshFilter()
            image_with_facemesh = facemesh_filter.detect_and_draw_facemesh(image)
            st.image(image_with_facemesh, channels="BGR", caption="Facemesh Filter Applied")

        elif filter_option == "Custom Makeup":
            if landmarks:
                st.sidebar.title("Customize Regions")

                # Define region colors, intensities, and controls
                region_colors = {}
                region_opacities = {}
                default_colors = {
                    "BLUSH_LEFT": "#FFCCCC",
                    "BLUSH_RIGHT": "#FFCCCC",
                    "EYES": "#00FF00",
                    "EYEBROWS": "#654321",
                    "EYESHADOW_LEFT": "#800080",
                    "EYESHADOW_RIGHT": "#800080",
                    "FACE": "#FFE0BD",
                    "LIP_UPPER": "#FF0000",
                    "LIP_LOWER": "#FF0000"
                }
                default_opacities = {
                    "BLUSH_LEFT": 0.3,
                    "BLUSH_RIGHT": 0.3,
                    "EYES": 0.5,
                    "EYEBROWS": 0.5,
                    "EYESHADOW_LEFT": 0.4,
                    "EYESHADOW_RIGHT": 0.4,
                    "FACE": 0.2,
                    "LIP_UPPER": 0.6,
                    "LIP_LOWER": 0.6,
                }

                reset = st.sidebar.button("Reset to Default Colors and Intensity")
                for region_name, default_color in default_colors.items():
                    if reset:
                        color = default_color
                        intensity = default_opacities[region_name]
                    else:
                        color = st.sidebar.color_picker(f"{region_name} Color", default_color)
                        intensity = st.sidebar.slider(f"{region_name} Intensity", 0.1, 1.0, default_opacities[region_name])

                    # Convert RGB to BGR and store opacity
                    rgb_color = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convert to BGR
                    region_colors[region_name] = bgr_color
                    region_opacities[region_name] = intensity

                try:
                    image_with_makeup = apply_makeup(image, landmarks, REGIONS, region_colors, region_opacities)
                    st.image(image_with_makeup, channels="BGR", caption="Custom Makeup Applied")
                except Exception as e:
                    st.error(f"Error applying makeup: {e}")
            else:
                st.warning("No landmarks detected. Please upload a clear image.")


if __name__ == "__main__":
    main()
