import argparse
import signal
import os
import time

from pytorch.network import ConvolutionalNeuralNetwork
from pytorch.network import LinearNeuralNetwork
from lir2.lir2 import LIR2
from lir2.lir2_renderer import Renderer

_PROGRAM_DESCRIPTION = '''
The program classifies the "images" captured by the sensor using a neural network model.
The program can be closed with SIGINT (ctrl+c on windows)

Requirements:
    - pytorch: https://pytorch.org/get-started/locally/
    - opencv: https://pypi.org/project/opencv-python/
    - numpy: https://pypi.org/project/numpy/
    - minimalmodbus: https://pypi.org/project/minimalmodbus/
    - serial: https://pypi.org/project/pyserial/'''

def handler(_0, _1):
    Renderer.close_window()
    exit(0)

if __name__ == '__main__':
    #Signal handler for interupt signal (CTRL+C on windows)
    signal.signal(signal.SIGINT, handler)

    parser = argparse.ArgumentParser(prog='Classificator', description=_PROGRAM_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='Path to the file containint the model')
    parser.add_argument('-s', '--serial', help='Serial port onto which the sensor is connected', default='COM3')
    parser.add_argument('-d', '--display', help='Display the thermal image on the screen', action='store_true', default=False)
    parser.add_argument('-m', '--model', help='Model to be used for classification', choices=['convolutional', 'linear'], default='convolutional')

    args = parser.parse_args()
    config = vars(args)

    source = config['source']
    serial = config['serial']
    display = bool(config['display'])
    model = config['model']

    if not os.path.exists(source):
        print('The provided source file does not exist')
        exit(1)

    n_net = None
    if model == 'convolutional':
        n_net = ConvolutionalNeuralNetwork()
    elif model == 'linear':
        n_net = LinearNeuralNetwork()
    else:
        print('Invlaid model')
        exit(1)

    sensor = LIR2(serial, 234)
    n_net.load_model(source)
    while True:
        start = time.time()
        mat = sensor.read_samples()
        if display:
            image = Renderer.render(mat)
            Renderer.display(image)

        classification = bool(n_net.predict(mat))
        print(f'\rPerson present: {classification} ', end='')

        duration = time.time() - start
        time.sleep(max(0, 1 - duration))
