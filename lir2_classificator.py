import pickle
import Orange

with open('./Orange/SVM(RAW).pkcls', 'rb') as model:
    lr = pickle.load(model)

