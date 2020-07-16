#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm,preprocessing 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates


# In[2]:


#Read Data
prices = pd.read_csv("data.csv",parse_dates=['date'])
prices


# In[3]:


fig, ax = plt.subplots(figsize=(6, 6))

# Add x-axis and y-axis
ax.plot(prices['date'],prices['close'],color='blue')


# In[4]:


# dropping unused columns 
prices.drop(["time", "open", "high", "low", "volume"],axis = 1, inplace = True) 

# Store the original dates for plotting the predictions
copy_df = prices.copy()
#copy_df = copy_df.reset_index()

#convert date to ints
prices['date'] = prices['date'].map(mdates.date2num)

# display 
copy_df


# In[5]:


#Make dataframe for test set's date
arr_date = copy_df['date'].to_list()
idx_test_date = int(0.75*len(arr_date))
df = pd.DataFrame(columns = ['test_date']) 
df['test_date'] = copy_df['date'].iloc[idx_test_date:]
df


# In[6]:


#Convert data date to list
arr = prices['date'].to_list()
x = [[]]
for j in range(len(arr)):
    arr_sem = []
    arr_sem.append(arr[j])
    x.append(arr_sem)

x.pop(0)
print(x)

#Convert data close price to list
arr_close = prices['close'].to_list()
y = [[]]
for j in range(len(arr_close)):
    arr_sem = []
    arr_sem.append(arr_close[j])
    y.append(arr_sem)
y.pop(0)
print(y)

print(len(x))    
print(len(y))


# In[7]:


#Data scalling using MinMax Scaler for date
scaler = preprocessing.MinMaxScaler()
x_scaledprice = scaler.fit_transform(x)
print(x_scaledprice)

#Data scalling using MinMax Scaler for close price
scaler = preprocessing.MinMaxScaler()
y_scaledprice = scaler.fit_transform(y)
print(y_scaledprice)

#Convert 2d array y to 1d
y = np.array(y_scaledprice)
y_price = y.flatten()

#split data to 75% for train data and 25% for test data 
x_train,x_test,y_train,y_test = train_test_split(x_scaledprice,y_price,test_size=0.25,random_state=None, shuffle=False)


# In[10]:


#SVR using kernels
for kernel_arg in ['rbf','poly','linear']:
    if (kernel_arg == 'rbf'):
        clf = svm.SVR(kernel ='rbf',C=1e3,gamma=0.1)
    elif (kernel_arg == 'poly'):
        clf = svm.SVR(kernel ='poly',C=1e3,degree=3)
    else:
        clf = svm.SVR(kernel ='linear',C=1e3)
    y_predict = clf.fit(x_train,y_train).predict(x_test)

    print('The MSE of %s : %f'%(kernel_arg, mean_squared_error(y_test, y_predict)))
    
    #plot data
    plt.subplots(figsize=(8, 6))
    plt.plot(df['test_date'], y_test, label = "real")
    plt.plot(df['test_date'], y_predict, label = "predict")
    plt.legend(bbox_to_anchor=(1.04,1))
    plt.show()


# In[ ]:





# In[ ]:




