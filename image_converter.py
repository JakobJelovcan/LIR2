from lir2.lir2_renderer import Renderer
import argparse
import csv
import cv2
import numpy as np
import os

_PROGRAM_DESCRIPTION = '''
The program converts the raw data stored in the csv file in to images.
Requirements:
    - opencv: https://pypi.org/project/opencv-python/
    - numpy: https://pypi.org/project/numpy/
    - minimalmodbus: https://pypi.org/project/minimalmodbus/
    - serial: https://pypi.org/project/pyserial/'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Image converter', description=_PROGRAM_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', help='Path to the CSV file containing data from the LIR2 sensor')
    parser.add_argument('-f', '--format', help='An integer representing a cv2 colormap (-1 for grayscale) https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html')
    parser.add_argument('-d', '--directory', help='Path to the directory in which the images should be stored')

    args = parser.parse_args()
    config = vars(args)
    colormap = cv2.COLORMAP_INFERNO if config['format'] is None else int(config['format'])

    directory = config['directory'] if config['directory'] is not None else '.'

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(config['file']) as file:
        reader = csv.reader(file)
        next(reader, None)
        for i, row in enumerate(reader):
            data = [float(v) for v in row[0:-1]]
            person = row[-1]
            matrix = np.array(np.split(np.array(data), 12))
            image = Renderer.render(matrix, colormap)
            cv2.imwrite(os.path.join(directory, f'ThermalImage{i}-Person({person}).jpg'), image)
