from torch import device
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Flatten
from torch.nn import Sequential
from torch.nn import Sigmoid


class ConvolutionalNetwork(Module):
    def __init__(self, device:device, hidden_channels=24):
        super().__init__()
        #Layer sizes are marked as (N, C, H, W)
        #N: batch size
        #C: number of channels. Number of channels on the input of the first layer is 1 (one color), the rest are determined by the hidden_channels parameter
        #H: height
        #W: width

        #Layer1
        #Input -> (?, 1, 12, 16)
        #Output -> (?, ?, 6, 8)
        self.layer1 = Sequential(
                        Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device),
                        ReLU(),
                        Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device),
                        ReLU(),
                        MaxPool2d(kernel_size=2, stride=2))
        #Layer2
        #Input -> (?, ?, 6, 8)
        #Output -> (?, ?, 3, 4)
        self.layer2 = Sequential(
                        Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device),
                        ReLU(),
                        Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device),
                        ReLU(),
                        MaxPool2d(2))
        
        #Layer3
        #Input -> (?, ?, 3, 4)
        #Output -> float
        self.layer3 = Sequential(
                        Flatten(),
                        Linear(in_features=3*4*hidden_channels, out_features=1, device=device),
                        Sigmoid())
        

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        return output
    
class LinearNetwork(Module):
    def __init__(self, device:device):
        super().__init__()

        self.net = Sequential(
                        Linear(in_features=192, out_features=100, device=device),
                        ReLU(),
                        Linear(in_features=100, out_features=100, device=device),
                        ReLU(),
                        Linear(in_features=100, out_features=1, device=device),
                        Sigmoid())
        
    def forward(self, input):
        return self.net(input)
