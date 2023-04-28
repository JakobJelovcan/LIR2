import numpy as np
import cv2

ZOOM_FACTOR = 40.0
WINDOW_NAME = 'Thermal sensor'

class Renderer:

    def render(matrix:np.ndarray, colormap:int=cv2.COLORMAP_INFERNO) -> np.ndarray:
        '''Transform the 16x12 matrix of temperatures in to an image.
        The matrix is scaled by a factor of 40 using a Lanczos upscaler, normalized and then converted to color using the specified colormap (Inferno by default)

        Parameters:
            matrix (numpy.ndarray): 16x12 matrix of temperatures to be converted
            colormap (int): an integer representing a cv2 colormap (-1 for grayscale)
        Returns:
            image (numpy.ndarray): cv2 image in the form of a numpy matrix representing the picture from the sensor'''
        
        scaled_image = cv2.resize(matrix, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LANCZOS4)
        matrix_shape = matrix.shape
        image_size = (int(matrix_shape[0] * ZOOM_FACTOR), int(matrix_shape[1] * ZOOM_FACTOR))
        matrix_u8 = np.zeros(image_size, dtype=np.uint8)
        matrix_u8 = cv2.normalize(scaled_image, matrix_u8, 0, 255, cv2.NORM_MINMAX)
        matrix_u8 = np.uint8(matrix_u8)
        return matrix_u8 if colormap == -1 else cv2.applyColorMap(matrix_u8, colormap)
    
    def display(image:np.ndarray) -> int:
        '''Display the image on the screen, and returns the pressed key

        Parameters:
            image (numpy.ndarray): image to be displayed
        Returns:
            pressed_key (int): the key that was pressed
        '''
        cv2.imshow(WINDOW_NAME, image)
        return cv2.waitKey(1)
    
    def close_window() -> None:
        '''Closes all open cv2 windows
        
        Parameters:
        Returns:
            None
        '''
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def is_window_open() -> bool:
        '''Checks if the window is open

        Parameters:
        Returns:
            open (bool): boolean indicating if the window is open
        '''
        return cv2.getWindowProperty(WINDOW_NAME, 0) != -1