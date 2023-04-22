from numpy import vstack
from torch import from_numpy
from torch import float
from torch import device
from torch.utils.data import DataLoader
from torch.cuda import is_available
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score

from pytorch.network import LinearNetwork
from pytorch.network import ConvolutionalNetwork
from pytorch.dataset import CSVDataset
from pytorch.dataset import CSVDatasetLinear
from pytorch.dataset import CSVDatasetConvolutional

import torch.jit as jit
import numpy as np

class Learner:
    def __init__(self):
        self.device = device('cuda:0') if is_available() else device('cpu')
        self.train_dl = None
        self.test_dl = None
        self.model = None

    def load_data(self, dataset:CSVDataset, batch_size:int, split:float) -> None:
        (train, test) = dataset.get_splits(split)
        self.train_dl = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    def train(self, epoch_count:int=300, learning_rate:float=0.0001, weight_decay:float=0.0001, debug:bool=False) -> None:
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
        (predictions, actuals) = (list(), list())
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
        self.model = jit.load(path)
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

class LinearLearner(Learner):
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
        self.model = LinearNetwork(self.device)
        super().train(epoch_count, learning_rate, weight_decay, debug)

    def predict(self, mat : np.ndarray) -> int:
        '''Predict the class from the input data
        Parameters:
            mat (ndarray): matrix containing the data
        Returns:
            class (int): integer representing the predicted class
        '''
        data = from_numpy(mat.reshape(1, -1)).type(float).to(self.device)
        return self.model(data).cpu().detach().numpy().round()[0]


class ConvolutionalLearner(Learner):
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
        self.model = ConvolutionalNetwork(self.device, 32)
        super().train(epoch_count, learning_rate, weight_decay, debug)

    def predict(self, mat : np.ndarray) -> int:
        '''Predict the class from the input data
        Parameters:
            mat (ndarray): matrix containing the data
        Returns:
            class (int): integer representing the predicted class
        '''
        #Reshape the matrix to (1, 1, 12, 16)
        mat.shape = (1,1, ) + mat.shape
        data = from_numpy(mat).type(float).to(self.device)
        return self.model(data).cpu().detach().numpy().round()[0]