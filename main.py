import lir2
from lir2_renderer import Renderer
import yolo_classificator
import time

if __name__ == '__main__':
    try:
        lir2 = lir2.LIR2('COM3', 234)
        yolo = yolo_classificator.YoloClassificator('./YOLO/yolov3.cfg', './YOLO/yolov3.weights', './YOLO/yolov3.names')

        repeat = True
        while repeat:
            start = time.time()
            mat = lir2.read_samples()
            image = Renderer.render(mat)
            Renderer.display(image)
            repeat = Renderer.is_window_open()
            time.sleep(max(1.0, 1 - (time.time() - start))) #Execute the loop every second

    except Exception as e:
        print(e)