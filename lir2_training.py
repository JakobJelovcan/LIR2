import lir2
from lir2_renderer import Renderer
import yolo_classificator
import cv2
import csv
import signal
import time
import os

FILE_PATH = './data/training_data.csv'

def handler(_0, _1):
    save_training_data(FILE_PATH, training_data)
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

    sensor = lir2.LIR2('COM3', 234)
    yolo = yolo_classificator.YoloClassificator('./yolo/yolov3.cfg', './yolo/yolov3.weights', './yolo/yolov3.names')
    camera = cv2.VideoCapture(0)

    training_data = []

    i = 1
    while True:
        matrix = sensor.read_samples()
        time.sleep(1)
        (_, camera_image) = camera.read()
        objects = yolo.classify(camera_image)
        training_data.append((matrix, objects))
        image = Renderer.render(matrix)
        Renderer.display(image)
        print(f"Frame: {i}, person: {'person' in objects}")
        i += 1