#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np


# In[61]:


data=pd.read_csv("iris.csv")
data


# In[66]:


data.info()


# In[74]:


data.dtypes


# In[77]:


data.describe()


# In[88]:


x=data['SepalLengthCm'].to_numpy()
print("Min:",x.min(),"\nMax:",x.max(),"\nStd:",x.std())


# In[68]:


plt.hist(x=data['Species'])
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.title('Count of different species')
plt.show()


# In[85]:


data.hist(figsize=(10,10))


# In[86]:


data.plot.box()


# In[44]:


data.boxplot(figsize=(10,5))


# In[40]:


sbn.pairplot(data)


# In[43]:


data['Species'].value_counts()


# In[48]:


sbn.boxplot(x=data['Species'],y=data['SepalLengthCm'])


# In[ ]:





# In[ ]:





# In[ ]:




