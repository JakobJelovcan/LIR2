import cv2
import csv
import signal
import time
import os
import argparse

from lir2.lir2 import LIR2
from lir2.lir2_renderer import Renderer
from yolo.yolo_classificator import YoloClassificator

_PROGRAM_DESCRIPTION = '''The program collects the data from the sensor and extends it with the classification aquired from the image.
                         The image from the camera is classified using the YOLO classification model.
                         A camera is required for this program to work'''

output_file_path = None
training_data = []

def handler(_0, _1):
    save_training_data(output_file_path, training_data)
    Renderer.close_window()
    exit(0)

def save_training_data(path:str, data:list) -> None:
    '''Saves the training data to the specified file. If the file already exists the new data is appended to it

    Parameters:
        path (str): path to the file
        data (list): a list of data to be written to the file
    Returns:
        None
    '''
    if os.path.exists(path):
        with open(path, 'a') as file:
            write_to_csv(csv.writer(file, lineterminator='\n'), data)
    else:
        with open(path, 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow([f"x{x}y{y}" for y in range(12) for x in range(16)] + ['person'])
            write_to_csv(writer, data)

def write_to_csv(writer, data:list) -> None:
    '''Writes the data into a csv file using the csv._writer

    Parameters:
        writer (csv.writer): writer object used for writing
        data (list): a list of data to be written
    Returns:
        None
    '''
    for (matrix, objects) in data:
        writer.writerow(list(matrix.flatten()) + [str('person' in objects)])

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)

    parser = argparse.ArgumentParser(prog='Trainer', description=_PROGRAM_DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--serial', help='Serial port onto which the sensor is connected', default='COM3')
    parser.add_argument('-d', '--display', help='Display the thermal image on the screen', action='store_true', default=False)
    parser.add_argument('-f', '--file', help='Path to the file in which the data should be stored (if the file exists the data will be appended to the bottom, default: data/training_data.csv)', default='./data/training_data.csv')
    parser.add_argument('-c', '--camera', help='Index of the camera to be used for image classification', default=0)

    args = parser.parse_args()
    config = vars(args)

    serial = config['serial']
    display = bool(config['display'])
    output_file_path = config['file']
    cam_index = int(config['camera'])

    sensor = LIR2(serial, 234)
    camera = cv2.VideoCapture(cam_index)

    if not os.path.exists('./yolo/yolov3.cfg'):
        print('YOLO config file is missing.\n it can be downloaded at https://pjreddie.com/darknet/yolo/ (YOLOv3 416)')
        exit(1)

    if not os.path.exists('./yolo/yolov3.weights'):
        print('YOLO weights file is missing.\n it can be downloaded at https://pjreddie.com/darknet/yolo/ (YOLOv3 416)')
        exit(1)

    if not os.path.exists('./yolo/yolov3.names'):
        print('YOLO names file is missing.\n it can be downloaded at https://github.com/pjreddie/darknet/blob/master/data/coco.names')
        exit(1)

    yolo = YoloClassificator('./yolo/yolov3.cfg', './yolo/yolov3.weights', './yolo/yolov3.names')

    while True:
        start = time.time()
        matrix = sensor.read_samples()
        (_, image) = camera.read()
        objects = yolo.classify(image)
        training_data.append((matrix, objects))
        if display:
            Renderer.display(Renderer.render(matrix))

        print(f'\rPerson present: {"person" in objects} ', end='')
        duration = time.time()
        time.sleep(max(0, 1 - duration))
        