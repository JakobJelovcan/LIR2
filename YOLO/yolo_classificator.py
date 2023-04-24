import cv2
import numpy as np


class YoloClassificator:
    def classify(self, image) -> str:
        blob = cv2.dnn.blobFromImage(
            image, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.layer_names)

        return {self.labels[np.argmax(detection[5:])] for output in layer_outputs for detection in output if (detection[5:][np.argmax(detection[5:])] > self.confidence)}

    def __init__(self, config: str, weights: str, names: str, confidence=0.5) -> None:
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        layer_names = self.net.getLayerNames()
        self.layer_names = [layer_names[i - 1]
                            for i in self.net.getUnconnectedOutLayers()]
        self.labels = open(names).read().strip().split('\n')
        self.confidence = confidence
