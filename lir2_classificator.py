from scikit_learn.network import Network
from pytorch.learning import ConvolutionalLearner
from pytorch.learning import LinearLearner
from lir2 import LIR2
from lir2_renderer import Renderer

if __name__ == '__main__':
    sensor = LIR2('COM5', 234)
    scikit_net = Network()
    conv_net = ConvolutionalLearner()
    lin_net = LinearLearner()
    scikit_net.load_model('./models/sklearn_model.pkcls')
    conv_net.load_model('./models/convolutional_model_cpu.pt')
    lin_net.load_model('./models/linear_model_cpu.pt')

    while True:
        mat = sensor.read_samples()
        image = Renderer.render(mat)
        Renderer.display(image)
        pred1 = bool(scikit_net.predict(mat))
        pred2 = bool(conv_net.predict(mat))
        pred3 = bool(lin_net.predict(mat))
        print(f'Person:\n\tNeural network: {pred1}\n\tConvolutional network: {pred2}\n\tLinear network: {pred3}')
