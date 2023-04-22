import lir2
from lir2_renderer import Renderer
import yolo_classificator
import cv2
import csv
import signal
import time

def handler(_0, _1):
    save_to_csv(training_data)
    exit(0)

def save_to_csv(training_data):
    header = [f"x{x}y{y}" for y in range(12) for x in range(16)] + ["person"]
    with open('./Data/new_training_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for (matrix, objects) in training_data:
            array = matrix.flatten()
            person = "person" in objects
            writer.writerow(list(array) + [str(person)])

if __name__ == '__main__':

    signal.signal(signal.SIGINT, handler)

    sensor = lir2.LIR2('COM3', 234)
    yolo = yolo_classificator.YoloClassificator('./YOLO/yolov3.cfg', './YOLO/yolov3.weights', './YOLO/yolov3.names')
    camera = cv2.VideoCapture(0)

    training_data = []

    i = 1
    while True:
        matrix = sensor.read_samples()
        time.sleep(.8)
        (_, camera_image) = camera.read()
        objects = yolo.classify(camera_image)
        training_data.append((matrix, objects))
        rendered_image = Renderer.render(matrix)
        #cv2.imwrite(f'./Images/ThermalImage{i}-Person({str("person" in objects)}).jpg', rendered_image)
        Renderer.display(rendered_image)
        print(f"Frame: {i}, person: {'person' in objects}")
        i += 1