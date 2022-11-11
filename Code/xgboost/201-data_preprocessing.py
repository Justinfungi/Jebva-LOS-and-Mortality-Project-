import pandas as pd
import numpy as np
import os 
from sklearn import preprocessing 

## set the path of the data
root = os.getcwd()
path_X = os.path.join(root, 'X')
path_y_t2 = os.path.join(root, 'Task2')

## read data 
X_train = pd.read_csv(os.path.join(path_X, 'X_train.csv'), index_col=[0], header=[0, 1, 2])
X_valid = pd.read_csv(os.path.join(path_X, 'X_valid.csv'), index_col=[0], header=[0, 1, 2])
X_test = pd.read_csv(os.path.join(path_X, 'X_test.csv'), index_col=[0], header=[0, 1, 2])

## clean y data
y_train_t2 = pd.read_csv(os.path.join(path_y_t2, 'Y_train.csv'))
y_valid_t2 = pd.read_csv(os.path.join(path_y_t2, 'Y_valid.csv'))

y_train_t2.set_index('Unnamed: 0', inplace=True) # set the id column as index
y_valid_t2.set_index('Unnamed: 0', inplace=True)

scaler = preprocessing.StandardScaler()

X_train_norm = pd.DataFrame(scaler.fit_transform(X_train.values)) # normalize the data
X_train_norm.columns = X_train.columns.droplevel(1) # drop column head "Aggregation Function" <- we use mean only
X_train = X_train_norm.iloc[:, 47:7488:72] # select means only

X_valid_norm = pd.DataFrame(scaler.transform(X_valid.values))
X_valid_norm.columns = X_valid.columns.droplevel(1) 
X_valid = X_valid_norm.iloc[:, 47:7488:72]

X_test_norm = pd.DataFrame(scaler.transform(X_test.values))
X_test_norm.columns = X_test.columns.droplevel(1) 
X_test = X_test_norm.iloc[:, 47:7488:72] # select means only


