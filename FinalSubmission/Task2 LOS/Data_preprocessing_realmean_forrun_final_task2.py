#!/usr/bin/env python
# coding: utf-8

# # Libraries

# ! pip install numpy pandas sklearn

# In[2]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")


# # Import data

# ## X

# In[3]:


# For training the model
X_train_realmean = pd.read_csv("../X/Xtrainmean.csv", index_col=[0])

# For cross validation
X_valid_realmean = pd.read_csv("../X/Xvalidmean.csv", index_col=[0])

# For prediction
X_test_realmean = pd.read_csv("../X/Xtestmean.csv", index_col=[0])


# In[4]:


X_train_realmean


# ## Task 2

# In[5]:


y_train_t2 = pd.read_csv("../Task2/Y_train.csv",index_col=[0])
y_valid_t2 = pd.read_csv("../Task2/Y_valid.csv",index_col=[0])


# In[6]:


y_train_t2_value=y_train_t2["los_icu"]
y_valid_t2_value=y_valid_t2["los_icu"]


# # Process 1. Raw Real Mean

# In[7]:


X_feature=X_train_realmean.join(y_train_t2)


# ### Process 1.1 Important Raw Real Mean Pairplot Visualization

# In[8]:


X_feature_plot=pd.DataFrame(X_feature[["diastolic blood pressure","heart rate","temperature","glascow coma scale total","los_icu"]])
X_feature_0=[]
X_feature_1=[]

for i in range(len(X_feature_plot)):
    if X_feature.iloc[i,-1]==0:
        X_feature_0.append(X_feature_plot.iloc[i,:])
    else:
        X_feature_1.append(X_feature_plot.iloc[i,:])


# In[9]:


import seaborn as sns
sns.pairplot(X_feature_plot,kind="scatter",hue="los_icu",plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))


# ## Process 2: Elimination of features with variance<0.08 (call is"nozero") and Imputation

# In[10]:


train=X_train_realmean.join(y_train_t2)
valid=X_valid_realmean.join(y_valid_t2)
test=X_test_realmean


# In[11]:


# eliminate feautres with variance  < 0.08
train_low_var = train.var()[train.var()<0.08].index[:-1]
test.drop(train_low_var,axis = 1,inplace=True)
train.drop(train_low_var,axis = 1,inplace=True)
valid.drop(train_low_var,axis = 1,inplace=True)

X_train_nozero = train.copy()
headnozero=X_train_nozero.columns
X_valid_nozero = valid.copy()
X_test_nozero  =test.copy()


# In[12]:


# Impute the 0 with mean
imp = SimpleImputer(missing_values=0, strategy='mean')
X_train_nozero = pd.DataFrame(imp.fit_transform(X_train_nozero))
X_train_nozero.columns=headnozero
X_train_nozero.index=X_train_realmean.index
X_valid_nozero = pd.DataFrame(imp.fit_transform(X_valid_nozero))
X_valid_nozero.columns=headnozero
X_test_nozero = pd.DataFrame(imp.fit_transform(X_test_nozero))
X_test_nozero.columns=headnozero[:-1]
X_train_nozero


# In[13]:


X_train_selected_t2=X_train_nozero
X_valid_selected_t2=X_valid_nozero
X_test_selected_t2=X_test_nozero


# ## Process 3: Normalization ##

# In[20]:


scaler = preprocessing.StandardScaler()

X_train_nozero_selected=X_train_nozero.values
X_train_nozero_scaled = scaler.fit_transform(X_train_nozero_selected)
X_train_selected_t2_norm=pd.DataFrame(X_train_nozero_scaled)
X_train_selected_t2_norm.columns=headnozero
X_train_selected_t2_norm.index=X_train_realmean.index
X_train_selected_t2_norm=X_train_selected_t2_norm.drop(columns=["los_icu"])

X_valid_nozero_selected=X_valid_nozero.values
X_valid_nozero_scaled = scaler.fit_transform(X_valid_nozero_selected)
X_valid_selected_t2_norm=pd.DataFrame(X_valid_nozero_scaled)
X_valid_selected_t2_norm.columns=headnozero
X_valid_selected_t2_norm=X_valid_selected_t2_norm.drop(columns=["los_icu"])


X_test_nozero_selected=X_test_nozero.values
X_test_nozero_scaled = scaler.fit_transform(X_test_nozero_selected)
X_test_selected_t2_norm=pd.DataFrame(X_test_nozero_scaled)
X_test_selected_t2_norm.columns=headnozero[:-1]


# In[21]:


X_train_selected_t2_norm


# # Data report

# In[15]:


print("*"*60)
print("There are 3 set of X for task 2")
print("X_train_selected_t2, X_valid_selected_t2,X_test_selected_t2")
print("-"*60)
print("Normalized version")
print("X_train_selected_t2_norm,X_valid_selected_t2_norm,X_test_selected_t2_norm")
print("-"*60)
print("There are 2 set of Y for task 2")
print("y_train_t2, y_valid_t2")
print("when training, please use: 'y_train_t2_value,y_valid_t2_value'")
print("*"*60)

