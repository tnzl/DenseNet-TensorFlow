# Imports
from tensorflow.data import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_data(dataset):
    if dataset == 'cifar100':
        from tensorflow.keras.datasets import cifar100
        (x_tr, y_tr), (x_te, y_te) = cifar100.load_data()
    elif dataset == 'cifar10':
        from tensorflow.keras.datasets import cifar10
        (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
    
    preprocesses = ([todtype, normalize], [ohe])
    
    x_te, y_te = preprocess(x_te, y_te, preprocesses)
    x_tr, y_tr = preprocess(x_tr, y_tr, preprocesses)
    
    tr_ds_x = Dataset.from_tensor_slices(x_tr)  
    tr_ds_y = Dataset.from_tensor_slices(y_tr)         
    te_ds_x = Dataset.from_tensor_slices(x_te)         
    te_ds_y = Dataset.from_tensor_slices(y_te)
    
    tr_ds = Dataset.zip((tr_ds_x, tr_ds_y)).shuffle(1000).batch(128)
    te_ds = Dataset.zip((te_ds_x, te_ds_y)).batch(128)

    return tr_ds, te_ds




def normalize(x, var = None):
    min_val = np.min(x)
    max_val = np.max(x)
    return (x-min_val) / (max_val-min_val)

def normalize_elemental(x):
    # element wise normalize 
    raise('Not yet implemented!')

def ohe(y):
    return OneHotEncoder().fit_transform(np.array(y).reshape(-1,1)).toarray()

def todtype(x, dtype='float32'):
    return x.astype(dtype)

def preprocess(x, y, preprocesses):
    
    for process in preprocesses[0]:
        x = process(x)

    for process in preprocesses[1]:
        y = process(y)
    
    return x, y 