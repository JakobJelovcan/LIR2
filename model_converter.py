import torch
import argparse
import os
from pytorch.network import ConvolutionalNeuralNetwork
from pytorch.network import LinearNeuralNetwork

_PROGRAM_DESCRIPTION = '''
The program converts amodel from TorchScript to onnx format
Requirements:
    - pytorch: https://pytorch.org/get-started/locally/
    - onnx: https://pypi.org/project/onnx/
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='Path to the source model file (.pt)')
    parser.add_argument('output', help='Path to the output file')
    parser.add_argument('-m', '--model', help='Model to convert', choices=['convolutional', 'linear'], default='convolutional')
    parser.add_argument('-f', '--format', help='Format of the converted model', choices=['onnx'], default='onnx')

    args = parser.parse_args()
    config = vars(args)

    source = config['source']
    if not os.path.exists(source):
        print('The provided source file does not exist')
        exit(1)

    output = config['output']
    model = config['model']
    format = config['format']

    n_net = None
    X = None
    if model == 'convolutional':
        n_net = ConvolutionalNeuralNetwork()
        X = torch.randn(size=(1, 1, 12, 16))
    elif model == 'linear':
        n_net = LinearNeuralNetwork()
        X = torch.rand(size=(1, 192))
    else:
        print('Invalid model')
        exit(1)

    n_net.load_model(source)

    if format == 'onnx':
        torch.onnx.export(n_net.model,
                        X.to(n_net.device),
                        output,
                        export_params=True,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': 
                                            {0: 'batch_size'},
                                        'output':
                                            {0: 'batch_size'}
                                        })