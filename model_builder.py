import argparse
from os import path

from pytorch.network import ConvolutionalNeuralNetwork
from pytorch.network import LinearNeuralNetwork

_PROGRAM_DESCRIPTION = '''
The program trains and stores a neural network model using the raw data from the sensor stored in a csv file.
Requirements:
    - pytorch: https://pytorch.org/get-started/locally/
    - opencv: https://pypi.org/project/opencv-python/
    - numpy: https://pypi.org/project/numpy/
    - minimalmodbus: https://pypi.org/project/minimalmodbus/
    - serial: https://pypi.org/project/pyserial/'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser( prog='Model builder', description=_PROGRAM_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='Path to the csv file containint the training data')
    parser.add_argument('-o', '--output', help='Path to the file into which the trained model shuild be stored')
    parser.add_argument('-s', '--split', help='Share of the cases to be used as test data (default: 0)', default=0)
    parser.add_argument('-a', '--accuracy', help='Print accuracy result (split has to be more than 0)', action='store_true', default=False)
    parser.add_argument('-b', '--batch', help='Batch size to be used for training (default: 512)', default=512)
    parser.add_argument('-e', '--epoch', help='Epoch count to be used for training (default: 300)', default=300)
    parser.add_argument('-m', '--model', help='Model to be trained (convolutional/linear)', choices=['convolutional', 'linear'], default='convolutional')

    args = parser.parse_args()
    config = vars(args)

    source = config['source']
    if not path.exists(source):
        print('The provided source file does not exist')
        exit(1)

    output = config['output'] if config['output'] is not None else 'pytorch_model.pt'
    split = float(config['split'])
    accuracy = bool(config['accuracy'])
    batch_size = int(config['batch'])
    epoch_count = int(config['epoch'])
    model = str(config['model'])

    n_net = None
    if model == 'convolutional':
        n_net = ConvolutionalNeuralNetwork()
    elif model == 'linear':
        n_net = LinearNeuralNetwork()
    else:
        print('Invalid model')
        exit(1)

    print('Loading data ...')
    n_net.load_data(source, batch_size, split)
    print('Training model ...')
    n_net.train(epoch_count, info=True)
    print()
    print('Storing model ...')
    n_net.store_model(output)
    if accuracy:
        print(f'Classification accuracy: {n_net.test()}')
