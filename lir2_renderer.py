import numpy as np
import cv2

ZOOM_FACTOR = 40.0
WINDOW_NAME = 'Thermal sensor'

class Renderer:

    def render(matrix, colormap = cv2.COLORMAP_INFERNO):
        '''Transform the 16x12 matrix of temperatures in to an image.
        The matrix is scaled by a factor of 40 using a Lanczos upscaler, normalized and then converted to color using the specified colormap (Inferno by default)

        Parameters:
            matrix (numpy.matrix): 16x12 matrix of temperatures to be converted
            colormap (int): an integer representing a cv2 colormap (-1 for grayscale)
        Returns:
            image (numpy.matrix): cv2 image in the form of a numpy matrix representing the picture from the sensor'''
        
        scaled_image = cv2.resize(matrix, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LANCZOS4)
        matrix_shape = matrix.shape
        image_size = (int(matrix_shape[0] * ZOOM_FACTOR), int(matrix_shape[1] * ZOOM_FACTOR))
        matrix_u8 = np.zeros(image_size, dtype=np.uint8)
        matrix_u8 = cv2.normalize(scaled_image, matrix_u8, 0, 255, cv2.NORM_MINMAX)
        matrix_u8 = np.uint8(matrix_u8)
        return matrix_u8 if colormap == -1 else cv2.applyColorMap(matrix_u8, colormap)

    def display(image):
        cv2.imshow(WINDOW_NAME, image)
        return cv2.waitKey(1)
    
    def close_window():
        cv2.destroyAllWindows()

    def is_window_open():
        return cv2.getWindowProperty(WINDOW_NAME, 0) != -1