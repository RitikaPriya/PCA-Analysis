#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import random as rd


# In[4]:


from sklearn.decomposition import PCA
from sklearn import preprocessing
#The preprocessing package gives us functions for scaling the data before performing PCA


# In[5]:


import matplotlib.pyplot as plt


# In[7]:


genes = ['gene' + str(i) for i in range(1,101)]


# In[8]:


wt = ['wt' + str(i) for i in range(1,6)]


# In[9]:


ko = ['ko' + str(i) for i in range(1,6)]


# In[10]:


Data = pd.DataFrame(columns=[*wt, *ko], index=genes)
#stars unpack the 'wt' and 'ko' arrays so that colum names are single arrays otherwise we would get array of 2 arrays and that wouodn't create 12 columns


# In[29]:


Data.loc[genes,'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
Data.loc[genes,'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)


# In[30]:


print(Data.head())


# In[31]:


print(Data.shape)


# # PCA

# In[32]:


#We need to centre and scale the data first
#After centering, average value for each gene will be 0, after scaling sd will be 1.
#The scale function expects the data to be in rows 
scaled_data = preprocessing.scale(Data.T)
# or StandardScaler().fit_transform(data.T)


# In[33]:


#Create PCA object. sklearn uses objects that can be trained using one dataset and applied to another dataset
pca = PCA()
pca.fit(scaled_data) #We do pca math. Calculate loading scores and variation each principle component accounts for


# In[34]:


pca_data = pca.transform(scaled_data)
#To draw the graph, we get coordinates through this


# In[35]:


per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
# percentage of variation of each principle component 
print(per_var)


# In[36]:


labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]


# # Scree plot

# In[37]:


plt.bar(x=range(1,len(per_var)+1), height = per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principle Component')
plt.title('Scree Plot')
plt.show()


# # PCA plot

# In[39]:


pca_df = pd.DataFrame(pca_data, index=[*wt,*ko], columns=labels)
#putting new coordinates created by pca.transform into a nice matrix 


# In[47]:


plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('PCA plot')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
#this adds sample names to graph
plt.show()


# # Loading scores

# In[48]:


loading_scores = pd.Series(pca.components_[0], index=genes)


# In[49]:


sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
#sorting loading scores based on their magnitudes


# In[51]:


top_10_genes = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_genes])


# In[ ]:




