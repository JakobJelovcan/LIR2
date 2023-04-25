from numpy import vstack
from torch import inference_mode
from torch import from_numpy
from torch import float
from torch import device
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Flatten
from torch.nn import Sequential
from torch.nn import Sigmoid
from torch.nn import BCELoss
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score

from pytorch.dataset import _CSVDataset
from pytorch.dataset import CSVDatasetLinear
from pytorch.dataset import CSVDatasetConvolutional

import torch.jit as jit
import numpy as np



class _ConvolutionalModel(Module):
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
    
class _LinearModel(Module):
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

class _NeuralNetwork:
    def __init__(self):
        self.device = device('cuda:0') if is_available() else device('cpu')
        self.train_dl = None
        self.test_dl = None
        self.model = None

    def load_data(self, dataset:_CSVDataset, batch_size:int, split:float) -> None:
        (train, test) = dataset.get_splits(split)
        self.train_dl = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    def train(self, epoch_count:int=300, learning_rate:float=0.0001, weight_decay:float=0.0001, debug:bool=False) -> None:
        self.model.train()
        criterion = BCELoss().to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(epoch_count):
            if debug:
                print(f'Epoch: {epoch}')

            for inputs, targets in self.train_dl:
                pred = self.model(inputs)
                loss = criterion(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self) -> float:
        '''Runs the model on test data and calculates the accuracy
        
        Parameters:
        Returns:
            accuracy (float): classification accuracy of the model
        '''
        self.model.eval()
        (predictions, actuals) = (list(), list())
        with inference_mode():
            for inputs, targets in self.test_dl:
                predicted = self.model(inputs).cpu().detach().numpy().round()
                actual = targets.cpu().numpy()
                actual = actual.reshape((len(actual), 1))
                predictions.append(predicted)
                actuals.append(actual)
            (predictions, actuals) = (vstack(predictions), vstack(actuals))
        return accuracy_score(actuals, predictions)
    
    def load_model(self, path : str) -> None:
        '''Loads the model from a TorchScript file
            
        Parameters:
            path (str): path to the file containing the model
        Returns:
            None
        '''
        self.model = jit.load(path, map_location=self.device)
        self.model.eval()
    
    def store_model(self, path : str) -> None:
        '''Stores the model in to a TorchScript file

        Parameters:
            path (str): path to the file into which the model should be stored
        Returns:
            None
        '''
        scripted = jit.script(self.model)
        scripted.save(path)

class LinearNeuralNetwork(_NeuralNetwork):
    def __init__(self):
        super().__init__()

    def load_data(self, path:str, batch_size:int=512, split:float=0.2) -> None:
        '''Loads the data from the csv file into a Dataset object

        Parameters:
            path (str): path to the csv file containing the data
            batch_size (int): the size of the batch to be used by the model
            split (float): the percentage of cases to be used as test data
        Returns:
            None
        '''
        dataset = CSVDatasetLinear(path, self.device)
        super().load_data(dataset, batch_size, split)

    def train(self, epoch_count:int=300, learning_rate:float=0.0001, weight_decay:float=0.0001, debug:bool=False) -> None:
        '''Creates a new linear model and trains it on the provided training data. The criterion function used is BCELoss and the optimizer used is Adam

        Parameters:
            epoch_count (int): the number of iterations used in the training
            learning_rate (float): learning rate passed on to the Adam optimizer
            weight_decay (float): weight decay passed on to the Adam optimizer
            debug (bool): enables the printout of the current epoch
        Returns:
            None
        '''
        self.model = _LinearModel(self.device)
        super().train(epoch_count, learning_rate, weight_decay, debug)

    def predict(self, mat : np.ndarray) -> int:
        '''Predict the class from the input data
        Parameters:
            mat (ndarray): matrix containing the data
        Returns:
            class (int): integer representing the predicted class
        '''
        self.model.eval()
        with inference_mode():
            data = from_numpy(mat.reshape(1, -1)).type(float).to(self.device)
            accuracy = self.model(data).cpu().detach().numpy().round()[0]
        return accuracy


class ConvolutionalNeuralNetwork(_NeuralNetwork):
    def __init__(self):
        super().__init__()

    def load_data(self, path:str, batch_size:int=512, split:float=0.2) -> None:
        '''Loads the data from the csv file into a Dataset object

        Parameters:
            path (str): path to the csv file containing the data
            batch_size (int): the size of the batch to be used by the model
            split (float): the percentage of cases to be used as test data
        Returns:
            None
        '''
        dataset = CSVDatasetConvolutional(path, self.device)
        super().load_data(dataset, batch_size, split)

    def train(self, epoch_count:int=300, learning_rate:float=0.0001, weight_decay:float=0.0001, debug:bool=False) -> None:
        '''Creates a new convolutional model and trains it on the provided training data. The criterion function used is BCELoss and the optimizer used is Adam

        Parameters:
            epoch_count (int): the number of iterations used in the training
            learning_rate (float): learning rate passed on to the Adam optimizer
            weight_decay (float): weight decay passed on to the Adam optimizer
            debug (bool): enables the printout of the current epoch
        Returns:
            None
        '''
        self.model = _ConvolutionalModel(self.device, 48)
        super().train(epoch_count, learning_rate, weight_decay, debug)

    def predict(self, mat : np.ndarray) -> int:
        '''Predict the class from the input data
        Parameters:
            mat (ndarray): matrix containing the data
        Returns:
            class (int): integer representing the predicted class
        '''
        #Reshape the matrix to (1, 1, 12, 16)
        self.model.eval()
        with inference_mode():
            mat.shape = (1,1, ) + mat.shape
            data = from_numpy(mat).type(float).to(self.device)
            accuracy = self.model(data).cpu().detach().numpy().round()[0]
        return accuracy
