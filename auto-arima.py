# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:40:32 2020

@author: RADHIKA
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


# Import CSV file into a dataframe
delhidata=pd.read_excel('D:\DS project Files\Delhi (1).xlsx')
#EDA    
#Index(['date', 'pm25'], dtype='object')
delhidata.head()
delhidata=delhidata.iloc[::-1]
delhidata.head()
delhidata.info()
delhidata.dtypes
delhidata['pm25']  = pd.to_numeric(delhidata['pm25'] ,errors='coerce')
delhidata.dtypes
delhidata.sort_values("date", axis = 0, ascending = True,inplace = True, na_position ='last')
delhidata1 = pd.DataFrame({'date': pd.date_range('2018-01-01', '2018-04-21', freq='1H', closed='left')})
delhidata2 = delhidata1.iloc[:2617,:]
delhidata3 = pd.merge(delhidata,delhidata2,on='date',how='right') 
delhidata3.info()
delhidata3.sort_values("date", axis = 0, ascending = True,inplace = True, na_position ='last')
sns.heatmap(delhidata.isnull(),cbar=True)
delhidata3.tail()
delhidata3.isna().sum()
delhidata3.info()
delhidata3.set_index(['date'],inplace=True)
delhidata3.shape
delhidata3.isnull().sum()
delhidata3_linear=delhidata3.interpolate(method='linear')
delhidata3_linear.isnull().sum()
delhidata3_linear.plot()
delhidata3_linear.shape
delhidata3_linear.plot(figsize=(15,3), color="blue", title='DELHI AIR QUALITY')
delhidata3_linear.hist()
delhidata3_linear.shape
delhidata3_linear.head()
from numpy import log
X = delhidata3_linear.values
X = log(X)
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))#mean1=5.335263, mean2=4.597500
print('variance1=%f, variance2=%f' % (var1, var2))#variance1=0.519288, variance2=0.700707
#ADF test
from statsmodels.tsa.stattools import adfuller

X = delhidata3_linear.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])#-4.057066
print('p-value: %f' % result[1])# 0.001139
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#1%: -3.433
#5%: -2.863
#10%: -2.567
#Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary  
    
    
expWeightedAverage=delhidata3_linear.ewm(halflife=12,min_periods=0,adjust=True).mean()
expWeightedstd=delhidata3_linear.ewm(halflife=12,min_periods=0,adjust=True).std()
plt.plot(delhidata3_linear)
plt.plot(expWeightedAverage,color='red')
plt.plot(expWeightedstd,color='black')    

logscale_weg=delhidata3_linear-expWeightedAverage
logscale_weg.head()
logscale_weg=logscale_weg[1:]
logscale_weg.head()
logscale_weg=pd.DataFrame(logscale_weg)
logscale_weg.head()
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(logscale_weg)
plt.show()
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(logscale_weg,lags=12)
tsa_plots.plot_pacf(logscale_weg,lags=12)


####auto-arima

train = logscale_weg[:int(0.7*(len(logscale_weg)))]
test = logscale_weg[int(0.7*(len(logscale_weg))):]
train#1831
test#785
train['pm25'].plot()
test['pm25'].plot()

#building the model
from pmdarima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(test, label='test')
plt.plot(forecast, label='Prediction')
plt.legend(loc='best')
plt.show()

#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error
auto_rmse = np.sqrt(mean_squared_error(test,forecast))
print(auto_rmse)
#79.17029955369242

import pickle
pickle.dump(model,open('model1.pkl','wb'))





