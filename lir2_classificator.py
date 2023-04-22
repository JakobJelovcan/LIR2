from scikit_learn.network import Network
from pytorch.learning import ConvolutionalLearner
from pytorch.learning import LinearLearner
from lir2 import LIR2

if __name__ == '__main__':
    sensor = LIR2('COM3', 234)
    ck_net = Network()
    conv_net = ConvolutionalLearner()
    lin_net = LinearLearner()
    ck_net.load_model('./models/sklearn_model.pkcls')
    conv_net.load_model('./models/convolutional_model.pt')
    lin_net.load_model('./models/linear_model.pt')

    while True:
        mat = sensor.read_samples()
        pred1 = bool(ck_net.predict(mat))
        pred2 = bool(conv_net.predict(mat))
        pred3 = bool(lin_net.predict(mat))
        print(f'Person: {pred1}, {pred2}, {pred3}')
