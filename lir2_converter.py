import lir2_renderer
import argparse
import csv
import cv2
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
            person = bool(row[-1])
            matrix = np.array(np.split(np.array(data), 12))
            image = lir2_renderer.Renderer.render(matrix, colormap)
            cv2.imwrite(os.path.join(directory, f'ThermalImage{i}-Person({person}).jpg'), image)

            