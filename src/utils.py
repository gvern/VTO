import cv2
import csv
import numpy as np

def load_image_with_alpha(image_path):
    """Loads an image with an alpha channel."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Error: Cannot load image from path: {image_path}. Please check the file path or integrity.")
    
    # Check if the image has an alpha channel
    if img.shape[-1] == 4:  # Image with alpha channel
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))
    else:  # Image without alpha channel
        b, g, r = cv2.split(img)
        alpha = np.ones(img.shape[:2], dtype=img.dtype) * 255  # Create a fully opaque alpha channel

    return img, alpha

def load_annotations(csv_path):
    """Loads annotations from a CSV file."""
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            return [list(map(int, row)) for row in reader]
    except Exception as e:
        raise FileNotFoundError(f"Error loading annotations from {csv_path}: {e}")
