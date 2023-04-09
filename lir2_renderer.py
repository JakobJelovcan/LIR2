import numpy as np
from scipy import ndimage
import cv2
import lir2


class Renderer:
    ZOOM_FACTOR = 40.0
    WINDOW_NAME = 'Thermal sensor'

    def render(self, matrix):
        matrix_shape = matrix.shape
        scaled_matrix = ndimage.zoom(matrix, self.ZOOM_FACTOR)

        image_size = (int(matrix_shape[0] * self.ZOOM_FACTOR), int(matrix_shape[1] * self.ZOOM_FACTOR))
        matrix_u8 = np.zeros(image_size, dtype=np.uint8)
        matrix_u8 = cv2.normalize(scaled_matrix, matrix_u8, 0, 255, cv2.NORM_MINMAX)
        matrix_u8 = np.uint8(matrix_u8)
        return cv2.applyColorMap(matrix_u8, cv2.COLORMAP_INFERNO)


    def display(self, image):
        cv2.imshow(self.WINDOW_NAME, image)
        return cv2.waitKey(1)
    
    def close_window(self):
        cv2.destroyAllWindows()

    def is_window_open(self):
        return cv2.getWindowProperty(self.WINDOW_NAME, 0) != -1

    def __init__(self):
        cv2.startWindowThread()