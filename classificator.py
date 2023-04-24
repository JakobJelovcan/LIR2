import argparse
import signal
from os import path

from pytorch.network import ConvolutionalNeuralNetwork
from pytorch.network import LinearNeuralNetwork
from lir2.lir2 import LIR2
from lir2.lir2_renderer import Renderer

_PROGRAM_DESCRIPTION = '''The program classifies the "images" captured by the sensor using a neural network model.'''

def handler(_0, _1):
    Renderer.close_window()
    exit(0)

if __name__ == '__main__':
    #Signal handler for interupt signal (CTRL+C on windows)
    signal.signal(signal.SIGINT, handler)

    parser = argparse.ArgumentParser(prog='Classificator', description=_PROGRAM_DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source', help='Path to the file containint the model')
    parser.add_argument('-s', '--serial', help='Serial port onto which the sensor is connected', default='COM3')
    parser.add_argument('-d', '--display', help='Display the thermal image on the screen', action='store_true', default=False)
    parser.add_argument('-m', '--model', help='Model to be used for classification', choices=['convolutional', 'linear'])

    args = parser.parse_args()
    config = vars(args)

    source = config['source']
    serial = config['serial']
    display = bool(config['display'])
    model = config['model']

    if not path.exists(source):
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
        mat = sensor.read_samples()
        if display:
            image = Renderer.render(mat)
            Renderer.display(image)
        
        classification = n_net.predict(mat)
        print(f'Person present: {classification}')
