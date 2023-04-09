import lir2
import lir2_renderer
import yolo_classificator
import time

if __name__ == '__main__':
    try:
        lir2 = lir2.LIR2('COM3', 234)
        yolo = yolo_classificator.YoloClassificator('./YOLO/yolov3.cfg', './YOLO/yolov3.weights', './YOLO/yolov3.names')
        rend = lir2_renderer.Renderer()

        repeat = True
        while repeat:
            start = time.time()
            mat = lir2.read_samples()
            image = rend.render(mat)
            rend.display(image)
            repeat = rend.is_window_open()

            time.sleep(max(1.0, 1 - (time.time() - start))) #Execute the loop every second

    except Exception as e:
        print(e)