from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv

class CSVDataset(Dataset):
    def __init__(self, path):
        #Read the content of the csv file in to a pandoc dataframe
        df = read_csv(path)

        #Extract the 2D matrix values and convert them to float32 (numpy array)
        self.values = df.values[:, :-1].astype('float32')

        #Encode the target clases as numeric values and convert them to float32. Then reshape them in to a 2D matrix with one column
        self.classes = LabelEncoder().fit_transform(df.values[:, -1]).astype('float32').reshape((df.shape[0], 1))


    '''Returns the number of rows in the matrix

    Parameters:
    Returns:
        length (int): number of rows in the matrix
    '''
    def __len__(self) -> int:
        return len(self.values)
    

    '''Returns a list of two arrays; the selected row of values and it's corresponding class
    
    Parameters:
        index (int): index of the selected row
    Returns:
        item (list): a list containing the selected row of values and the corresponding class
    '''
    def __getitem__(self, index) -> list:
        return [self.values[index], self.classes[index]]
    

    '''Splits the dataset in to a training and testing set

    Parameters:
        split (float): share of the values to be used as the testing set
    Returns:
        sets (list): a list of two datasets first one for training and the second one for testing
    '''
    def get_splits(self, split=0.33) -> list:
        test_size = round(split * len(self))
        train_size = len(self) - test_size
        return random_split(self, [train_size, test_size])