#!/usr/bin/env python
# coding: utf-8

# # Libraries

# ! pip install numpy pandas sklearn

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")


# # Import data

# ## X

# In[15]:


# For training the model
X_train_realmean = pd.read_csv("../X/Xtrainmean.csv", index_col=[0])

# For cross validation
X_valid_realmean = pd.read_csv("../X/Xvalidmean.csv", index_col=[0])

# For prediction
X_test_realmean = pd.read_csv("../X/Xtestmean.csv", index_col=[0])


# ## Task 1

# In[3]:


y_train_t1 = pd.read_csv("../Task1/Y_train.csv")
y_valid_t1 = pd.read_csv("../Task1/Y_valid.csv")


# In[4]:


y_train_t1_value=y_train_t1["mort_icu"]
y_valid_t1_value=y_valid_t1["mort_icu"]


# ## Task 2

# In[5]:


y_train_t2 = pd.read_csv("../Task2/Y_train.csv")
y_valid_t2 = pd.read_csv("../Task2/Y_valid.csv")


# In[6]:


y_train_t2_value=y_train_t2["los_icu"]
y_valid_t2_value=y_valid_t2["los_icu"]


# # Data Pre-processing

# In[13]:


X_train_realmean


# ## Process 2: Elimination of features containing 70% 0 value (call is"nozero") and Imputation

# In[16]:


# Eliminate feature containing 70% 0 value
import copy
X_train_nozero=copy.deepcopy(X_train_realmean)
X_valid_nozero=copy.deepcopy(X_valid_realmean)
X_test_nozero=copy.deepcopy(X_test_realmean)
for i in X_train_realmean.columns:
    if (X_train_nozero[i] == 0).sum()> 12000:
        X_train_nozero.drop(i, axis=1, inplace=True)

headnozero=list(X_train_nozero.columns.values)
X_valid_nozero = X_valid_nozero[X_train_nozero.columns]
X_test_nozero = X_test_nozero[X_train_nozero.columns]


# In[17]:


# Impute the 0 with mean
imp = SimpleImputer(missing_values=0, strategy='mean')
X_train_nozero = pd.DataFrame(imp.fit_transform(X_train_nozero))
X_train_nozero.columns=headnozero
X_valid_nozero = pd.DataFrame(imp.fit_transform(X_valid_nozero))
X_valid_nozero.columns=headnozero
X_test_nozero = pd.DataFrame(imp.fit_transform(X_test_nozero))
X_test_nozero.columns=headnozero
X_train_nozero


# ## Process 3: Lasso feature selection

# In[18]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_nozero)
sel_ = SelectFromModel(
    LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=10))
sel_.fit(scaler.transform(X_train_nozero), y_train_t1_value)
X_train_selected = pd.DataFrame(sel_.transform(scaler.transform(X_train_nozero)))


# In[19]:


import matplotlib.pyplot as plt
train_log_scores=[]
test_log_scores=[]
for c in range(1,11,1):
    log_ = LogisticRegression(C=0.1*c, penalty='l1', solver='liblinear', random_state=3612)
    log_.fit(scaler.transform(X_train_nozero), y_train_t1_value)
    train_log_scores.append(log_.score(X_train_nozero, y_train_t1_value))
    test_log_scores.append(log_.score(X_valid_nozero, y_valid_t1_value))
plt.plot(train_log_scores, 'bo--')
plt.plot(test_log_scores, 'bo-')
plt.ylim(0.92, 0.95)
plt.legend(["log training score", "log valid score"])
plt.axvline(np.argmax(test_log_scores), linestyle="dotted", color="red")
plt.annotate(np.max(test_log_scores).round(4), (np.argmax(test_log_scores), np.max(test_log_scores)), xycoords="data",
                 xytext=(40, 20), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                 horizontalalignment="center", verticalalignment="top")
plt.show()


# In[20]:


train_log_scores=[]
test_log_scores=[]
for c in range(1,11,1):
    log_ = LogisticRegression(C=0.1*c, penalty='l1', solver='saga', random_state=3612)
    log_.fit(scaler.transform(X_train_nozero), y_train_t1_value)
    train_log_scores.append(log_.score(X_train_nozero, y_train_t1_value))
    test_log_scores.append(log_.score(X_valid_nozero, y_valid_t1_value))
plt.plot(train_log_scores, 'bo--')
plt.plot(test_log_scores, 'bo-')
plt.ylim(0.92, 0.95)
plt.legend(["log training score", "log valid score"])
plt.axvline(np.argmax(test_log_scores), linestyle="dotted", color="red")
plt.annotate(np.max(test_log_scores).round(4), (np.argmax(test_log_scores), np.max(test_log_scores)), xycoords="data",
                 xytext=(40, 20), textcoords="offset pixels", arrowprops=dict(facecolor="black", shrink=0.1), fontsize=10,
                 horizontalalignment="center", verticalalignment="top")
plt.show()


# In[21]:


scaler = StandardScaler()
scaler.fit(X_train_nozero)
log_=LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=3612)
sel_ = SelectFromModel(log_)
sel_.fit(scaler.transform(X_train_nozero), y_train_t1_value)
X_train_selected_t1 = pd.DataFrame(sel_.transform(scaler.transform(X_train_nozero)))
X_valid_selected_t1 = pd.DataFrame(sel_.transform(scaler.transform(X_valid_nozero)))
X_test_selected_t1 = pd.DataFrame(sel_.transform(scaler.transform(X_test_nozero)))
cols=sel_.get_support(indices=True)
headnozero_new=[]
for i in cols:
    headnozero_new.append(headnozero[i])
X_train_selected_t1.columns=headnozero_new
X_valid_selected_t1.columns=headnozero_new
X_test_selected_t1.columns=headnozero_new
X_train_selected_t1


# In[23]:


X_train_selected_t2=X_train_nozero
X_test_selected_t2=X_test_nozero
X_valid_selected_t2=X_valid_nozero


# ## Normalization

# In[24]:


scaler = preprocessing.StandardScaler()

X_train_selected_t1=X_train_selected_t1.values
X_train_selected_t1_scaled = scaler.fit_transform(X_train_selected_t1)
X_train_selected_t1_norm=pd.DataFrame(X_train_selected_t1_scaled)

X_train_selected_t2=X_train_selected_t2.values
X_train_selected_t2_scaled = scaler.fit_transform(X_train_selected_t2)
X_train_selected_t2_norm=pd.DataFrame(X_train_selected_t2_scaled)

X_valid_selected_t1=X_valid_selected_t1.values
X_valid_selected_t1_scaled = scaler.fit_transform(X_valid_selected_t1)
X_valid_selected_t1_norm=pd.DataFrame(X_valid_selected_t1_scaled)

X_valid_selected_t2=X_valid_selected_t2.values
X_valid_selected_t2_scaled = scaler.fit_transform(X_valid_selected_t2)
X_valid_selected_t2_norm=pd.DataFrame(X_valid_selected_t2_scaled)

X_test_selected_t1=X_test_selected_t1.values
X_test_selected_t1_scaled = scaler.fit_transform(X_test_selected_t1)
X_test_selected_t1_norm=pd.DataFrame(X_test_selected_t1_scaled)

X_test_selected_t2=X_test_selected_t2.values
X_test_selected_t2_scaled = scaler.fit_transform(X_test_selected_t2)
X_test_selected_t2_norm=pd.DataFrame(X_test_selected_t2_scaled)


# # Data report

# In[25]:


print("*"*60)
print("There are 6 set of X")
print("X_train_selected_t1, X_train_selected_t2, X_valid_selected_t1,X_valid_selected_t2,X_test_selected_t1,X_test_selected_t2")
print("-"*60)
print("Normalized version")
print("X_train_selected_t1_norm, X_train_selected_t2_norm, X_valid_selected_t1_norm,X_valid_selected_t2_norm,X_test_selected_t1_norm,X_test_selected_t2_norm")
print("-"*60)
print("There are 4 set of Y")
print("y_train_t1, y_train_t2, y_valid_t1, y_valid_t2")
print("when training, please use: 'y_train_t1_value,y_train_t2_value,y_valid_t1_value,y_valid_t2_value'")
print("*"*60)

