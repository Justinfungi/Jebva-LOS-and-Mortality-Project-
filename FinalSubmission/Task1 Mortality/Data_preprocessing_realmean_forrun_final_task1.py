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

# In[2]:


# For training the model
X_train_realmean = pd.read_csv("../X/Xtrainmean.csv", index_col=[0])

# For cross validation
X_valid_realmean = pd.read_csv("../X/Xvalidmean.csv", index_col=[0])

# For prediction
X_test_realmean = pd.read_csv("../X/Xtestmean.csv", index_col=[0])


# In[3]:


X_train_realmean


# ## Task 1

# In[4]:


y_train_t1 = pd.read_csv("../Task1/Y_train.csv",index_col=[0])
y_valid_t1 = pd.read_csv("../Task1/Y_valid.csv",index_col=[0])


# In[5]:


y_train_t1_value=y_train_t1["mort_icu"]
y_valid_t1_value=y_valid_t1["mort_icu"]


# # Process 1. Raw Real Mean

# In[6]:


X_feature=X_train_realmean.join(y_train_t1)


# ### Process 1.1 Important Raw Real Mean Pairplot Visualization

# In[7]:


X_feature_plot=pd.DataFrame(X_feature[["diastolic blood pressure","heart rate","temperature","glascow coma scale total","mort_icu"]])
X_feature_0=[]
X_feature_1=[]

for i in range(len(X_feature_plot)):
    if X_feature.iloc[i,-1]==0:
        X_feature_0.append(X_feature_plot.iloc[i,:])
    else:
        X_feature_1.append(X_feature_plot.iloc[i,:])


# In[8]:


import seaborn as sns
sns.pairplot(X_feature_plot,kind="scatter",hue="mort_icu",plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))


# ## Process 2: Elimination of features containing 70% 0 value (call is"nozero") and Imputation

# In[9]:


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


# In[10]:


# Impute the 0 with mean
imp = SimpleImputer(missing_values=0, strategy='mean')
X_train_nozero = pd.DataFrame(imp.fit_transform(X_train_nozero))
X_train_nozero.columns=headnozero
X_train_nozero.index=X_train_realmean.index
X_valid_nozero = pd.DataFrame(imp.fit_transform(X_valid_nozero))
X_valid_nozero.columns=headnozero
X_test_nozero = pd.DataFrame(imp.fit_transform(X_test_nozero))
X_test_nozero.columns=headnozero
X_train_nozero


# ## Process 3: Normalization ##

# In[11]:


scaler = preprocessing.StandardScaler()

X_train_nozero_selected=X_train_nozero.values
X_train_nozero_scaled = scaler.fit_transform(X_train_nozero_selected)
X_train_nozero_norm=pd.DataFrame(X_train_nozero_scaled)
X_train_nozero_norm.columns=headnozero
X_train_nozero_norm.index=X_train_realmean.index

X_valid_nozero_selected=X_valid_nozero.values
X_valid_nozero_scaled = scaler.fit_transform(X_valid_nozero_selected)
X_valid_nozero_norm=pd.DataFrame(X_valid_nozero_scaled)
X_valid_nozero_norm.columns=headnozero


X_test_nozero_selected=X_test_nozero.values
X_test_nozero_scaled = scaler.fit_transform(X_test_nozero_selected)
X_test_nozero_norm=pd.DataFrame(X_test_nozero_scaled)
X_test_nozero_norm.columns=headnozero


# ## Process 4: Imbalanced ##

# ### Process 4.1 Increase the result="1" samples in training set

# In[12]:


Full = pd.DataFrame(np.concatenate((X_train_nozero_norm,pd.DataFrame(y_train_t1_value)),axis=1))
Full

label0 = Full[Full[61]==0]
label1 = Full[Full[61]==1]
print(label0.shape,label1.shape)


# In[13]:


from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler()
x_over, y_over = oversample.fit_resample(X_train_nozero_norm, y_train_t1_value)

y_over = pd.DataFrame(y_over)
y_over.shape
x_over.shape


# In[14]:


Full = pd.DataFrame(np.concatenate((x_over,y_over),axis=1))
Full

label0 = Full[Full[61]==0]
label1 = Full[Full[61]==1]
print(label0.shape,label1.shape)


# In[15]:


X_train_t1_balanced=x_over
X_train_t1_balanced.columns=headnozero
y_train_t1_balanced = y_over
X_train_t1_balanced.index=y_train_t1_balanced.index
X_train_t1_balanced=X_train_t1_balanced.join(pd.DataFrame(y_train_t1_balanced))


# ### Process 4.2 Increase the result="1" samples in validation set

# In[16]:


Full = pd.DataFrame(np.concatenate((X_valid_nozero_norm,pd.DataFrame(y_valid_t1_value)),axis=1))
Full

label0 = Full[Full[61]==0]
label1 = Full[Full[61]==1]
print(label0.shape,label1.shape)


# In[17]:


from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler()
x_over, y_over = oversample.fit_resample(X_valid_nozero_norm, y_valid_t1_value)

y_over = pd.DataFrame(y_over)
y_over.shape
x_over.shape


# In[18]:


Full = pd.DataFrame(np.concatenate((x_over,y_over),axis=1))
Full

label0 = Full[Full[61]==0]
label1 = Full[Full[61]==1]
print(label0.shape,label1.shape)


# In[19]:


X_valid_t1_balanced=x_over
X_valid_t1_balanced.columns=headnozero
y_valid_t1_balanced = y_over
X_valid_t1_balanced.index=y_valid_t1_balanced.index
X_valid_t1_balanced=X_valid_t1_balanced.join(pd.DataFrame(y_valid_t1_balanced))


# ### Process 4.3 Important balanced data pairplot visualization

# In[20]:


X_train_t1_plot=pd.DataFrame(X_train_t1_balanced[["diastolic blood pressure","heart rate","temperature","glascow coma scale total","mort_icu"]])
X_train_t1_plot_0=[]
X_train_t1_plot_1=[]

for i in range(len(X_train_t1_balanced)):
    if X_train_t1_plot.loc[i,"mort_icu"]==0:
        X_train_t1_plot_0.append(X_train_t1_plot.iloc[i,:])
    else:
        X_train_t1_plot_1.append(X_train_t1_plot.iloc[i,:])


# In[21]:


sns.pairplot(X_train_t1_balanced.iloc[:,-5:],kind="scatter",hue="mort_icu",plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))


# ## Process 5: Lasso feature selection

# In[22]:


from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import     KFold, RepeatedKFold, GridSearchCV,     cross_validate, train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[24]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=3612)
lasso_alphas = np.linspace(0, 0.2, 21)
lasso = Lasso()
grid = dict()
grid['alpha'] = lasso_alphas
gscv = GridSearchCV(     lasso, grid, scoring='roc_auc',     cv=cv, n_jobs=-1)
results = gscv.fit(X_train_t1_balanced.iloc[:,:-1], y_train_t1_balanced)
print('MAE: %.5f' % results.best_score_)
print('Config: %s' % results.best_params_)


# In[25]:


scaler = StandardScaler()
scaler.fit(X_train_t1_balanced.iloc[:,:-1])
log_=LogisticRegression(C=0.01, penalty='l1', solver='liblinear', random_state=3612)
sel_ = SelectFromModel(log_)
sel_.fit(scaler.transform(X_train_t1_balanced.iloc[:,:-1]), y_train_t1_balanced)
X_train_selected_t1 = pd.DataFrame(sel_.transform(scaler.transform(X_train_t1_balanced.iloc[:,:-1])))
X_valid_selected_t1 = pd.DataFrame(sel_.transform(scaler.transform(X_valid_t1_balanced.iloc[:,:-1])))
X_test_selected_t1 = pd.DataFrame(sel_.transform(scaler.transform(X_test_nozero_norm)))
cols=sel_.get_support(indices=True)
headnozero_new=[]
for i in cols:
    headnozero_new.append(headnozero[i])
X_train_selected_t1.columns=headnozero_new
X_valid_selected_t1.columns=headnozero_new
X_test_selected_t1.columns=headnozero_new
X_train_selected_t1


# In[26]:


X_train_selected_t1_paint=X_train_selected_t1.join(y_train_t1_balanced)


# In[27]:


sns.pairplot(X_train_selected_t1_paint.iloc[:,-4:],kind="scatter",hue="mort_icu",plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))


# In[28]:


X_train_selected_t1_plot=pd.DataFrame(X_train_t1_balanced[["diastolic blood pressure","heart rate","temperature","glascow coma scale total","mort_icu"]])
sns.pairplot(X_train_selected_t1_plot,kind="scatter",hue="mort_icu",plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))


# ## Normalization

# In[29]:


scaler = preprocessing.StandardScaler()

X_train_selected_t1=X_train_selected_t1.values
X_train_selected_t1_scaled = scaler.fit_transform(X_train_selected_t1)
X_train_selected_t1_norm=pd.DataFrame(X_train_selected_t1_scaled)


X_valid_selected_t1=X_valid_selected_t1.values
X_valid_selected_t1_scaled = scaler.fit_transform(X_valid_selected_t1)
X_valid_selected_t1_norm=pd.DataFrame(X_valid_selected_t1_scaled)


X_test_selected_t1=X_test_selected_t1.values
X_test_selected_t1_scaled = scaler.fit_transform(X_test_selected_t1)
X_test_selected_t1_norm=pd.DataFrame(X_test_selected_t1_scaled)


# # Data report

# In[32]:


print("*"*60)
print("There are 3 set of X")
print("X_train_selected_t1,X_valid_selected_t1,X_test_selected_t1")
print("-"*60)
print("Normalized version")
print("X_train_selected_t1_norm,X_valid_selected_t1_norm,X_test_selected_t1_norm")
print("-"*60)
print("There are 2 set of Y")
print("y_train_t1,y_valid_t1")
print("when training, please use: 'y_train_t1_value,y_valid_t1_value'")
print("*"*60)

