from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy

import numpy as np
import torch
from torch import nn

from pytorch.dataset import _CSVDataset
from pytorch.dataset import CSVDatasetLinear
from pytorch.dataset import CSVDatasetConvolutional

import torch.jit as jit
import numpy as np



class _ConvolutionalModel(nn.Module):
    def __init__(self, device:torch.device, hidden_channels=24):
        super().__init__()
        #Layer sizes are marked as (N, C, H, W)
        #N: batch size
        #C: number of channels. Number of channels on the input of the first layer is 1 (one color), the rest are determined by the hidden_channels parameter
        #H: height
        #W: width

        #Layer1
        #Input -> (?, 1, 12, 16)
        #Output -> (?, ?, 6, 8)
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.act11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.act12 = nn.ReLU()
        self.pool11 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Layer2
        #Input -> (?, ?, 6, 8)
        #Output -> (?, ?, 3, 4)
        self.conv21= nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.act21 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.act22 = nn.ReLU()
        self.pool21 = nn.MaxPool2d(kernel_size=2)
        
        #Layer3
        #Input -> (?, ?, 3, 4)
        #Output -> float

        self.fl31 = nn.Flatten()
        self.l31 = nn.Linear(in_features=3*4*hidden_channels, out_features=1, device=device)
        

    def forward(self, input):
        #Layer1 input (?, 1, 12, 16) -> Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d -> (?, ?, 6, 8)
        output = self.conv11(input)
        output = self.act11(output)
        output = self.conv12(output)
        output = self.act12(output)
        output = self.pool11(output)
        #Layer2 (?, ?, 6, 8) -> Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d -> (?, ?, 3, 4)
        output = self.conv21(output)
        output = self.act21(output)
        output = self.conv22(output)
        output = self.act22(output)
        output = self.pool21(output)
        #Layer3 (?, ?, 3, 4) -> Flatten -> Linear -> Sigmoid -> (1)
        output = self.fl31(output)
        output = self.l31(output)
        return output
    
class _LinearModel(nn.Module):
    def __init__(self, device:torch.device):
        super().__init__()

        self.l1 = nn.Linear(in_features=192, out_features=100, device=device)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(in_features=100, out_features=100, device=device)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(in_features=100, out_features=1, device=device)
        
    def forward(self, input):
        output = self.l1(input)
        output = self.act1(output)
        output = self.l2(output)
        output = self.act2(output)
        output = self.l3(output)
        return output

class _NeuralNetwork:
    def __init__(self):
        self.device = torch.device('cuda:0') if is_available() else torch.device('cpu')
        self.train_dl = None
        self.test_dl = None
        self.model = None

    def load_data(self, dataset:_CSVDataset, batch_size:int, split:float) -> None:
        (train, test) = dataset.get_splits(split)
        self.train_dl = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    def train(self, epoch_count:int=300, learning_rate:float=0.0001, weight_decay:float=0.0001, info:bool=False) -> None:
        self.model.train()
        loss_fn = nn.BCELoss().to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(epoch_count):
            if info:
                print(f'\r{round((epoch/epoch_count)*100)}%', end='')

            for X, y in self.train_dl:
                y_logits = self.model(X)
                y_prob = torch.sigmoid(y_logits)
                loss = loss_fn(y_prob, y)
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
        acc_fn = Accuracy(task='binary')
        (preds, actuals) = (list(), list())
        with torch.inference_mode():
            for X, y in self.test_dl:
                y_logits = self.model(X).squeeze().cpu()
                y_prob = torch.sigmoid(y_logits)
                y_pred = torch.round(y_prob)
                actual = y.squeeze().cpu()
                preds.append(y_pred)
                actuals.append(actual)
        return acc_fn(torch.hstack(preds), torch.hstack(actuals))
    
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

    def train(self, epoch_count:int=300, learning_rate:float=0.0001, weight_decay:float=0.0001, info:bool=False) -> None:
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
        super().train(epoch_count, learning_rate, weight_decay, info)

    def predict(self, mat : np.ndarray) -> int:
        '''Predict the class from the input data
        Parameters:
            mat (ndarray): matrix containing the data
        Returns:
            class (int): integer representing the predicted class
        '''
        self.model.eval()
        with torch.inference_mode():
            vector = mat.flatten()
            tensor = torch.tensor(vector, dtype=torch.float, device=self.device)
            prediction = self.model(tensor)
            prediction_cpu = prediction.cpu().detach()
        return prediction_cpu.numpy().round()[0]


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

    def train(self, epoch_count:int=300, learning_rate:float=0.0001, weight_decay:float=0.0001, info:bool=False) -> None:
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
        super().train(epoch_count, learning_rate, weight_decay, info)

    def predict(self, mat : np.ndarray) -> int:
        '''Predict the class from the input data
        Parameters:
            mat (ndarray): matrix containing the data
        Returns:
            class (int): integer representing the predicted class
        '''
        #Reshape the matrix to (1, 1, 12, 16)
        self.model.eval()
        with torch.inference_mode():
            tensor = torch.tensor(mat, dtype=torch.float, device=self.device)
            prediction = self.model(tensor.view((1,1,12,16)))
            prediction_cpu = prediction.cpu().detach()
        return prediction_cpu.numpy().round()[0]
