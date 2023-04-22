from sklearn.neural_network import MLPClassifier
from pandas import read_csv
from pickle import load
from pickle import dump

class Network:
    def __init__(self):
        self.model = None

    def load(self, path):
        '''Loads the model from a pickle file

        Parameters:
            path (str): path to the file in which the model is stored
        Returns:
            None
        '''
        with open(path, 'rb') as file:
            self.model = load(file)

    def store(self, path):
        '''Stores the model in to a pickle file

        Parameters:
            path (str): path to the file in to which the model should be stored
        Returns:
            None
        '''
        with open(path, 'wb') as file:
            dump(self.model, file)

    def train(self, path):
        '''Create and train the MLPClassifier using data in the csv file
            
        Parameters:
            path (str): path to the csv file containing the training data
        Returns:
            None
        '''
        self.model = MLPClassifier(solver='adam', alpha=0.0001, activation='relu')
        df = read_csv(path)
        x = df.values[:, :-1].astype('float32')
        y = df.values[:, -1].astype('float32')
        self.model.fit(x, y)

    def predict(self, vec):
        '''Predict the class from the input data
        Parameters:
            vec (array): vector containing the data
        Returns:
            class (array): array containing the predicted class
        '''
        return self.model.predict(vec)