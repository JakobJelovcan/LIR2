from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Module
from torch.nn.functional import relu

class Network(Module):
    def __init__(self):
        super(Network, self).__init__()

        #Layer 1
        self.conv1 = Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(12)

        #Layer 2
        self.conv2 = Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(12)

        #Layer 3
        self.pool = MaxPool2d(2, 2)

        #Layer 4
        self.conv4 = Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn4 = BatchNorm2d(24)

        #Layer 5
        self.conv5 = Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn5 = BatchNorm2d(24)

        #Layer 6
        self.fc1 = Linear(in_features=24*10*10, out_features=2)


    def forward(self, input):
        output = relu(self.bn1(self.conv1(input)))
        output = relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = relu(self.bn4(self.conv4(output)))
        output = relu(self.bn5(self.conv5(output)))
        output = self.fc1(output)
        return output
