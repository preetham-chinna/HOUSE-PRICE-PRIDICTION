#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


cd


# In[3]:


pwd


# In[4]:


x=pd.read_csv('Downloads/headbrain.csv')


# In[39]:


x


# In[20]:


plt.rcParams['figure.figsize']=(20.0,10.0)
data=pd.read_csv("Downloads/headbrain.csv")
print(data.shape)
data.head()
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
mean_x=np.mean(X)
mean_y=np.mean(Y)
n=len(X)
numer=0
denom=0
for i in range (n):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
    b1=numer/denom
    b0=mean_y-(b1*mean_x)
print(b1,b0)


# In[ ]:





# In[21]:


max_X=np.max(X)+100
min_X=np.min(X)-100
x=np.linspace(min_X,max_X,1000)
y=b0+b1*x
plt.plot(x,y,color='#58b970',label='Regline')
plt.scatter(X,Y,color='#ef5423',label='Scatplt')
plt.xlabel('HeadSize')
plt.ylabel('BrainHeight')
plt.legend()
plt.show()


# In[33]:


ss_t=0
ss_r=0
for i in range (n):
    y_pred=b0+b1*X[i]
    ss_t+=(Y[i]-mean_y)**2
    ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)


# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X=X.reshape((n,1))
reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)
mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
r2_score=reg.score(X,Y)
print(rmse)
print(r2_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




