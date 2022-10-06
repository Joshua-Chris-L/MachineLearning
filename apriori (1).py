#!/usr/bin/env python
# coding: utf-8

# # Apriori

# ## Importing the libraries

# In[1]:


get_ipython().system('pip install apyori')


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Data Preprocessing

# In[41]:


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = [ ]
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20) ])


# In[ ]:


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


# In[33]:


from apyori import apriori
rules = apriori(transactions = transactions, min_support =  0.003, min_confidence = 0.2, min_lift = 3, min_length = 2 , max_length = 2)


# In[34]:


results = list(rules)
results


# ## Visualising the results

# ### Displaying the first results coming directly from the output of the apriori function

# In[44]:


results = list(rules)


# In[ ]:


results


# ### Putting the results well organised into a Pandas DataFrame

# In[ ]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# ### Displaying the results non sorted

# In[ ]:


resultsinDataFrame


# ### Displaying the results sorted by descending lifts

# In[ ]:


resultsinDataFrame.nlargest(n = 10, columns = 'Lift')

