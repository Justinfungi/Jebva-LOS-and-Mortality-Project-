import pandas as pd
import os 
from sklearn import preprocessing

## set the path of the data
root = os.getcwd()
path_X = os.path.join(root, 'X')
path_y_t1 = os.path.join(root, 'Task1')
path_y_t2 = os.path.join(root, 'Task2')

## read data 
X_train = pd.read_csv(os.path.join(path_X, 'X_train.csv'), index_col=[0], header=[0, 1, 2])
X_valid = pd.read_csv(os.path.join(path_X, 'X_valid.csv'), index_col=[0], header=[0, 1, 2])
# X_test = pd.read_csv(os.path.join(path_X, 'X_test.csv'), index_col=[0], header=[0, 1, 2])

## clean y data
y_train_t1 = pd.read_csv(os.path.join(path_y_t1, 'Y_train.csv'), index_col=[0], header=[0])
y_valid_t1 = pd.read_csv(os.path.join(path_y_t1, 'Y_valid.csv'), index_col=[0], header=[0])

### concat X_train and X_valid
# print(X_train.shape, y_train_t1.shape)
# print(X_valid.shape, y_valid_t1.shape)

# X_train = pd.concat([X_train, X_valid], axis=0)
# y_train_t1 = pd.concat([y_train_t1, y_valid_t1], axis=0)

### Process 2: Elimination of features containing 70% 0 value (call is"nozero") and Imputation
# Eliminate feature containing 70% 0 value
# from sklearn.impute import SimpleImputer
# import copy
# X_train_nozero=copy.deepcopy(X_train)
# X_valid_nozero=copy.deepcopy(X_valid)
# X_test_nozero=copy.deepcopy(X_test)
# for i in X_train.columns:
#     if (X_train_nozero[i] == 0).sum()> 12000:
#         X_train_nozero.drop(i, axis=1, inplace=True)

# headnozero=list(X_train_nozero.columns.values)
# X_valid_nozero = X_valid_nozero[X_train_nozero.columns]
# X_test_nozero = X_test_nozero[X_train_nozero.columns]

# # Impute the 0 with mean
# imp = SimpleImputer(missing_values=0, strategy='mean')
# X_train_nozero = pd.DataFrame(imp.fit_transform(X_train_nozero))
# X_train_nozero.columns=headnozero
# X_valid_nozero = pd.DataFrame(imp.fit_transform(X_valid_nozero))
# X_valid_nozero.columns=headnozero
# X_test_nozero = pd.DataFrame(imp.fit_transform(X_test_nozero))
# X_test_nozero.columns=headnozero

## standardize x data
scaler = preprocessing.StandardScaler()

X_train_norm = pd.DataFrame(scaler.fit_transform(X_train.values)) # normalize the data
X_train_norm.columns = X_train.columns.droplevel([1]) # drop column head "Aggregation Function" <- we use mean only
X_train_t1 = X_train_norm.iloc[:, 47:7488:72] # select means only

X_valid_norm = pd.DataFrame(scaler.transform(X_valid.values))
X_valid_norm.columns = X_valid.columns.droplevel([1]) 
X_valid_t1 = X_valid_norm.iloc[:, 47:7488:72]

# X_test_norm = pd.DataFrame(scaler.transform(X_test.values))
# X_test_norm.columns = X_test.columns.droplevel([1, 2]) 
# X_test_t1 = X_test_norm.iloc[:, 47:7488:72]

# concat X_train and X_valid
# X_train_t1 = pd.concat([X_train_t1, X_valid_t1], axis=0)
# y_train_t1 = pd.concat([y_train_t1, y_valid_t1], axis=0)

print(X_train_t1.shape, y_train_t1.shape)
print(X_valid_t1.shape, y_valid_t1.shape)
