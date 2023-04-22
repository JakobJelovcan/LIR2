from scikit_learn.network import Network
from lir2 import LIR2

if __name__ == '__main__':
    sensor = LIR2('COM3', 234)
    net = Network()
    net.load('./Data/model.pkcls')

    while True:
        mat = sensor.read_samples()
        vec = mat.reshape(1, -1)
        pred = bool(net.predict(vec)[0])
        print(f'Person: {pred}')
