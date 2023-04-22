from torch.nn import Conv2d
from torch.nn import Sequential
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Module
from torch.nn.init import kaiming_uniform_

class Network(Module):
    def __init__(self):
        super(Network, self).__init__()

        self.layer1 = Sequential(
                        Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
                        ReLU(),
                        MaxPool2d(kernel_size=2, stride=2, padding=1))
        
        self.layer2 = Sequential(
                        Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                        ReLU(),
                        MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = Linear(3 * 4 * 4, 32, bias=True)
        kaiming_uniform_(self.fc1.weight)

        self.layer3 = Sequential(
                        self.fc1,
                        ReLU())
        
        self.fc2 = Linear(32, 1, bias=True)
        kaiming_uniform_(self.fc2.weight)

        self.layer4 = Sequential(
                        self.fc2,
                        ReLU())

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = output.view(output.size(0), -1)
        output = self.layer3(output)
        output = self.layer4(output)
        return output
