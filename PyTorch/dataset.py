import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv

class _CSVDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.X = None   #Matrix of n × 192 temperatures
        self.y = None   #Vector of classifications

    def __len__(self) -> int:
        '''Returns the number of rows in the matrix

        Parameters:
        Returns:
            length (int): number of rows in the matrix
        '''
        return len(self.X)
    

    def __getitem__(self, index) -> list:
        '''Returns a list of two arrays; the selected row of values and it's corresponding class
        
        Parameters:
            index (int): index of the selected row
        Returns:
            item (list): a list containing the selected row of values and the corresponding class
        '''
        return [self.X[index], self.y[index]]
    

    def get_splits(self, split=0.33) -> list:
        '''Splits the dataset in to a training and testing set

        Parameters:
            split (float): share of the values to be used as the testing set
        Returns:
            sets (list): a list of two datasets first one for training and the second one for testing
        '''
        test_size = round(split * len(self))
        train_size = len(self) - test_size
        return torch.utils.data.random_split(self, [train_size, test_size])

class CSVDatasetLinear(_CSVDataset):
    def __init__(self, path, device:torch.device):
        super().__init__()
        #Read the content of the csv file in to a pandoc dataframe
        df = read_csv(path)

        #Convert data to a tensor of floats on the specified device
        self.X = torch.from_numpy(df.values[:, :-1].astype('float32')).type(torch.float).to(device)

        #Encode the target clases as numeric values and convert them to a tensor of floats on the specified device
        y = LabelEncoder().fit_transform(df.values[:, -1]).reshape((len(df), 1))
        self.y = torch.from_numpy(y.astype('float32')).type(torch.float).to(device)

    
class CSVDatasetConvolutional(_CSVDataset):
    def __init__(self, path, device:torch.device):
        super().__init__()
        #Read the content of the csv file in to a pandoc dataframe
        df = read_csv(path)

        #Convert each row in to a 3D matrix of float32 values. Resulting matrix has a depth of one (one color channel), width of 16 and height of 12
        X = np.array([row.reshape((1, 12, 16)) for row in df.values[:, :-1]])
        self.X = torch.from_numpy(X.astype('float32')).type(torch.float).to(device)

        #Encode the target clases as numeric values and convert them to float32. Then reshape them in to a 2D matrix with one column
        y = LabelEncoder().fit_transform(df.values[:, -1])
        y = y.reshape((len(y), 1))
        self.y = torch.from_numpy(y.astype('float32')).type(torch.float).to(device)