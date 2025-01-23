import cv2
import numpy as np
from utils import load_annotations, load_image_with_alpha

class Filter:
    def __init__(self, filter_path, annotation_path):
        self.filter_path = filter_path
        self.annotation_path = annotation_path
        self.filter_image, self.alpha_channel = load_image_with_alpha(self.filter_path)
        self.annotations = load_annotations(self.annotation_path)

    def overlay_filter(self, frame, landmarks):
        """Overlays the filter on the frame."""
        points = [landmarks[idx] for idx in self.annotations]
        h, w = self.filter_image.shape[:2]

        # Define transformation matrix
        dst_points = np.array(points, dtype=np.float32)
        src_points = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)
        transform_matrix = cv2.getAffineTransform(src_points, dst_points[:3])

        # Warp the filter image
        warped_filter = cv2.warpAffine(self.filter_image, transform_matrix, (frame.shape[1], frame.shape[0]))
        warped_alpha = cv2.warpAffine(self.alpha_channel, transform_matrix, (frame.shape[1], frame.shape[0]))

        # Blend the warped filter with the frame
        overlay = cv2.addWeighted(frame, 1, warped_filter, 0.5, 0)
        return cv2.add(overlay, warped_alpha)
