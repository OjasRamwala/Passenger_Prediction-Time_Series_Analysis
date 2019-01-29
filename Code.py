
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6

dataset = pd.read_csv ('AirPassengers.csv')
# Parse strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'] , infer_datetime_format = True)
indexedDataset = dataset.set_index(['Month']) #setting index variable as month

'''#Visualising the data
from datetime import datetime
indexedDataset.tail(5)'''

plt.xlabel('Date')
plt.ylabel ('Number of Passengers')
plt.plot(indexedDataset)
# The dataset is non-stationary ... it has an upward trend

#Determining rolling statistics
rolmean = indexedDataset.rolling(window = 12).mean() #12 months->gives rolling mean at yearly level..Jan'1940 then Jan'1941

rolstd = indexedDataset.rolling (window = 12).std()
print (rolmean , rolstd)

# Plot rolling statistics 
orig = plt.plot(indexedDataset , color = 'orange' , label = 'Original')
mean = plt.plot(rolmean , color = 'black' , label = 'Rolling Mean')
std =  plt.plot(rolstd , color = 'blue' , label = 'Rolling Std')
plt.legend (loc = 'best')
plt.title ("Rolling Mean and Standard Deviation")
plt.show (block = False)

# Performing the Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(indexedDataset['#Passengers'] , autolag = 'AIC')
#The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection.
# AIC gives us the exact info of what we want in the time series
# Analyses the difference between the exact values and the estimated values
dfoutput = pd.Series (dftest[0:4] , index = ['Test Statistic' , 'p-value' , '#lags Used' , 'Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)' %key] = value
    
print(dfoutput)
#crtical value should be more than the Test Statistic-> Not satisfied here
#So non-stationary

# Estimating the trend
indexedDataset_logScale = np.log(indexedDataset)
plt.plot (indexedDataset_logScale)

movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingSTD = indexedDataset_logScale.rolling(window = 12).std()
plt.plot (indexedDataset_logScale)
plt.plot(movingAverage , color = 'red')
