import numpy as np
from scipy import ndimage
import cv2
import LIR2

ZOOM_FACTOR = 40.0

class Renderer:

    def render(self, matrix):
        scaled_matrix = ndimage.zoom(matrix, ZOOM_FACTOR)

        image_size = (int(self.height * ZOOM_FACTOR), int(self.width * ZOOM_FACTOR))
        matrix_u8 = np.zeros(image_size, dtype=np.uint8)
        matrix_u8 = cv2.normalize(scaled_matrix, matrix_u8, 0, 255, cv2.NORM_MINMAX)
        matrix_u8 = np.uint8(matrix_u8)
        image = cv2.applyColorMap(matrix_u8, cv2.COLORMAP_INFERNO)
        cv2.imshow('Thermal sensor', image)
        return cv2.waitKey(1)
    
    def close_window(self):
        cv2.destroyAllWindows()

    def __init__(self, sensor: LIR2):
        cv2.startWindowThread()
        self.sensor = sensor
        self.width = 16
        self.height = 12