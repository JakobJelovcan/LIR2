import time

import LIR2
import renderer
import cv2
import numpy as np

if __name__ == '__main__':
    try:
        lir2 = LIR2.LIR2('COM3', 234)
        render = renderer.Renderer(lir2)
        cam = cv2.VideoCapture(0)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        key = 0
        while key != 27:
            start = time.time()

            mat = lir2.read_samples()
            key = render.render(mat)
            
            res, image = cam.read()
            image = cv2.flip(image, 1, image)

            image = cv2.resize(image, (640, 480))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            boxes, weights = hog.detectMultiScale(image, winStride=(8,8))

            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

            for (xA, yA, xB, yB) in boxes:
                cv2.rectangle(image, (xA, yA), (xB, yB),
                                (0, 255, 0), 2)
            

            cv2.imshow("USB camera", image)
            cv2.waitKey(1)

            duration = time.time() - start
            time.sleep(1.0 - min(1.0, duration))

        render.close_window()

    except Exception as e:
        print(e)