#!/usr/bin/env python
# coding: utf-8

# # Random Forest Regression

# ## Importing the libraries

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[50]:


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# ## Training the Random Forest Regression model on the whole dataset

# In[51]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

#X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=15, random_state=0)
regr.fit(X, y)


# In[52]:


regr.predict([[6.5]])


# ## Predicting a new result

# In[54]:


regr.predict([[6.5]])


# ## Visualising the Random Forest Regression results (higher resolution)

# In[48]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




