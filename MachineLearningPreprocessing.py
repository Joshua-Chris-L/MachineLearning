#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing Tools

# ## Importing the libraries

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[4]:


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[5]:


print(X)


# In[7]:


print(y)


# ## Taking care of missing data

# In[8]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[9]:


print(X)


# ## Encoding categorical data

# ### Encoding the Independent Variable

# In[12]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[ ]:





# In[13]:


print(X)


# ### Encoding the Dependent Variable

# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[15]:


print(y)


# ## Splitting the dataset into the Training set and Test set

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[17]:


print(X_train)


# In[18]:


print(X_test)

print(y_train)

print(y_test)

# ## Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])


# In[17]:


print(X_train)


# In[18]:


print(X_test)

