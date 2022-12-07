## Libraries
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

## Import data

root = os.getcwd()
path_X = os.path.join(root, 'X')
path_y_t1 = os.path.join(root, 'Task1')
path_y_t2 = os.path.join(root, 'Task2')

## X

X_train = pd.read_csv(os.path.join(path_X, 'X_train.csv'), index_col=[0], header=[0, 1, 2])
X_valid = pd.read_csv(os.path.join(path_X, 'X_valid.csv'), index_col=[0], header=[0, 1, 2])
X_test = pd.read_csv(os.path.join(path_X, 'X_test.csv'), index_col=[0], header=[0, 1, 2])

# ## Task 1

y_train_t1 = pd.read_csv(os.path.join(path_y_t1, 'Y_train.csv'))
y_valid_t1 = pd.read_csv(os.path.join(path_y_t1, 'Y_valid.csv'))

y_train_t1.set_index('Unnamed: 0', inplace=True)
y_valid_t1.set_index('Unnamed: 0', inplace=True)

# ## Task 2

y_train_t2 = pd.read_csv(os.path.join(path_y_t2, 'Y_train.csv'))
y_valid_t2 = pd.read_csv(os.path.join(path_y_t2, "Y_valid.csv"))

y_train_t2.set_index('Unnamed: 0', inplace=True) # set the id column as index
y_valid_t2.set_index('Unnamed: 0', inplace=True)

# # Data Pre-processing

# ## Process 1: Mean
# 

li=[] # Find the positions of the columns with means of the particular feature after 24 hours
for i in range(47, 7488, 72):
    li.append(i)
#print(li)
headlist=list(X_train.columns.values)[1:]
head=[]
for i in li:
    head.append(headlist[i][0])

dx_train = pd. DataFrame(X_train)
X_train_mean= dx_train.iloc[:,li]
X_train_mean.columns=head

dx_valid = pd. DataFrame(X_valid)
X_valid_mean= dx_valid.iloc[:,li]
X_valid_mean.columns=head

dx_test = pd. DataFrame(X_test)
X_test_mean= dx_test.iloc[:,li]
X_test_mean.columns=head

# ## Process 2: Elimination of features containing 70% 0 value (call is"nozero") and Imputation

# Eliminate feature containing 70% 0 value
import copy
X_train_nozero=copy.deepcopy(X_train_mean)
X_valid_nozero=copy.deepcopy(X_valid_mean)
X_test_nozero=copy.deepcopy(X_test_mean)
for i in X_train_mean.columns:
    if (X_train_nozero[i] == 0).sum()> 12000:
        X_train_nozero.drop(i, axis=1, inplace=True)

headnozero=list(X_train_nozero.columns.values)
X_valid_nozero = X_valid_nozero[X_train_nozero.columns]
X_test_nozero = X_test_nozero[X_train_nozero.columns]

# Impute the 0 with mean
imp = SimpleImputer(missing_values=0, strategy='mean')
X_train_nozero = pd.DataFrame(imp.fit_transform(X_train_nozero))
X_train_nozero.columns=headnozero
X_valid_nozero = pd.DataFrame(imp.fit_transform(X_valid_nozero))
X_valid_nozero.columns=headnozero
X_test_nozero = pd.DataFrame(imp.fit_transform(X_test_nozero))
X_test_nozero.columns=headnozero

# ## Normalization
scaler = StandardScaler()

X_train_nozero = scaler.fit_transform(X_train_nozero)
X_train=pd.DataFrame(X_train_nozero)

X_valid_nozero = scaler.fit_transform(X_valid_nozero)
X_valid=pd.DataFrame(X_valid_nozero)

X_test_nozero = scaler.fit_transform(X_test_nozero)
X_test=pd.DataFrame(X_test_nozero)

# # Data report

# print("*"*60)
# print("There are 6 set of X")
# print("X_*_nozero")
# print("-"*60)
# print("Normalized version")
# print("X_*_norm")
# print("-"*60)
# print("There are 4 set of Y")
# print("y_train_t1, y_train_t2, y_valid_t1, y_valid_t2")
# print("when training, please use: 'y_train_t1_value,y_train_t2_value,y_valid_t1_value,y_valid_t2_value'")
# print("*"*60)

