from numpy import vstack
from pandas import read_csv
import numpy as np
from torch import cuda
from torch import device
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Conv2d
from torch.nn import Sequential
from torch.nn import Dropout
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import AvgPool2d
from torch.nn import Module
from torch import Tensor
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import sgd
from torch.nn import BCELoss
from torch.nn import BCEWithLogitsLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

KEEP_PROB = 0.95

class CSVDataset(Dataset):
    def __init__(self, path):
        df = read_csv(path)
        values = df.values[:, :-1].astype('float32')
        noise = np.random.normal(0, 1, values.shape).astype('float32')
        #self.X = np.concatenate((values, values + noise))
        self.X = Tensor(values).to(compute_device)
        self.y = df.values[:, -1]
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        #self.y = np.concatenate((self.y, self.y))
        self.y = Tensor(self.y.reshape((len(self.y), 1))).to(compute_device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return [self.X[i], self.y[i]]
    
    def get_splits(self, n_test=0.25):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])
    
class Network(Module):
    def __init__(self):
        super(Network, self).__init__()

        self.layer1 = Sequential(
                        Linear(in_features=192, out_features=100),
                        ReLU()).to(compute_device)
        
        self.layer1.apply(init_weights)
        
               
        self.layer3 = Sequential(
                        Linear(in_features=100, out_features=1),
                        Sigmoid()).to(compute_device)
        
        self.layer3.apply(init_weights)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer3(output)
        return output
    
def init_weights(m):
    if isinstance(m, Linear):
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    

def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=512, shuffle=True)
    test_dl = DataLoader(test, batch_size=512, shuffle=False)
    return train_dl, test_dl

def train_model(train_dl, model):
    criterion = BCELoss().to(compute_device)
    optimizer = Adam(model.parameters(), lr=0.1, weight_decay=0.0001)
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
        #print(f"Epoch: {epoch}")

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.cpu().detach().numpy()
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.round()
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    print(predictions)
    acc = accuracy_score(actuals, predictions)
    return acc

def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat

path = 'C:\\Users\\Jakob\\git\\LIR2\\Data\\training_data.csv'
compute_device = device('cuda:0') if cuda.is_available() else device('cpu')
train_dl, test_dl = prepare_data(path)
model = Network()
model.to(compute_device)
train_model(train_dl, model)
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)

# row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
# yhat = predict(row, model)
# print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))