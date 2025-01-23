import cv2
import csv
import numpy as np
from PIL import Image

def load_image_with_alpha(image_path):
    # Attempt to load the image
    try:
        img = Image.open("./filters/red_lipstick.png")
        img.show()  # Open the image to visually confirm it's correct
        print(f"Image format: {img.format}, Mode: {img.mode}")
    except Exception as e:
        print(f"Error loading image with PIL: {e}")
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(img.shape)
    if img is None:
        raise FileNotFoundError(f"Error: Cannot load image from path: {image_path}. Please check the file path or integrity.")
    
    # Check if the image has an alpha channel
    if img.shape[-1] == 4:  # Image with alpha channel
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))
    else:  # Image without alpha channel
        b, g, r = cv2.split(img)
        alpha = None  # No alpha channel

    return img, alpha

def load_annotations(csv_path):
    """Loads annotations from a CSV file."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        return [int(row[0]) for row in reader]
