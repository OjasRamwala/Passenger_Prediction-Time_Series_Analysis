import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6

dataset = pd.read_csv ('AirPassengers.csv')
# Parse strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'] , infer_datetime_format = True)
indexedDataset = dataset.set_index(['Month']) #setting index variable as month

'''
#Visualising the data
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

movingAverage = indexedDataset_logScale.rolling(window=12).mean() # window = 12 -> 12 months
movingSTD = indexedDataset_logScale.rolling(window = 12).std()
plt.plot (indexedDataset_logScale)
plt.plot(movingAverage , color = 'red')
# The plot shows that the mean is not stationary... it is still moving with time.. however in the log time it stil better than the previous one
#Upward Trend -> Data is not stationary

# We will get the difference between the moving average and the actual number of passengers 
datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage 
datasetLogScaleMinusMovingAverage.head(12) # Top 12 values

# Removing NaN (Not a Number) value
datasetLogScaleMinusMovingAverage.dropna(inplace = True)
datasetLogScaleMinusMovingAverage.head(10)

from statsmodels.tsa.stattools import adfuller
def test_stationarity (timeseries):
    
    #Deteriming the rolling statistics
    movingAverage = timeseries.rolling(window = 12).mean()
    movingSTD = timeseries.rolling(window = 12).std()
    
    #Plot rolling statistics 
    orig = plt.plot(timeseries , color = 'blue' , label = 'Original')
    mean = plt.plot(movingAverage , color = 'red' , label = 'Rolling Mean')
    std =  plt.plot(movingSTD , color = 'black' , label = 'Rolling Std')
    plt.legend (loc = 'best')
    plt.title ("Rolling Mean and Standard Deviation")
    plt.show (block = False)
    
    # Performing the Dickey-Fuller test
    from statsmodels.tsa.stattools import adfuller
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['#Passengers'] , autolag = 'AIC')
    #The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection.
    # AIC gives us the exact info of what we want in the time series
    # Analyses the difference between the exact values and the estimated values
    dfoutput = pd.Series (dftest[0:4] , index = ['Test Statistic' , 'p-value' , '#lags Used' , 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)
    
test_stationarity (datasetLogScaleMinusMovingAverage)    
                                                            '''Results of Dickey-Fuller Test:
                                                    Test Statistic                  -3.162908
                                                    p-value                          0.022235
                                                    #lags Used                      13.000000
                                                    Number of Observations Used    119.000000
                                                    Critical Value (1%)             -3.486535
                                                    Critical Value (5%)             -2.886151
                                                    Critical Value (10%)            -2.579896
                                                    dtype: float64'''
                                                    # P- Value is relatively less as compared to 0.99 obtained intially
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife = 12 , min_periods = 0 , adjust = True).mean()
plt.plot(indexedDataset_logScale)
plt.plot (exponentialDecayWeightedAverage) 
# As the time series is progressing the average is also progressing towards the higher side                                                   
                                                    
datasetLogScaleMinusExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
test_stationarity (datasetLogScaleMinusExponentialDecayAverage)
# Standard Deviation comes out to be flat...no trend
# Rolling mean has also improved 
                                                           '''Results of Dickey-Fuller Test:
                                                    Test Statistic                  -3.601262
                                                    p-value                          0.005737
                                                    #lags Used                      13.000000
                                                    Number of Observations Used    130.000000
                                                    Critical Value (1%)             -3.481682
                                                    Critical Value (5%)             -2.884042
                                                    Critical Value (10%)            -2.578770
                                                    dtype: float64'''
                                                
#Will shift the values into time series so that we can use for forecasting
datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)                                                        

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose (indexedDataset_logScale)
#seasonal decompose segregates 3 components - trend , seasonal , decomposition

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot (indexedDataset_logScale , label = 'Original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend , label = 'Trend')
plt.legend (loc = 'best')
plt.subplot(413)
plt.plot (seasonal , label = 'Seasonality')
plt.legend (loc = 'best')
plt.subplot (414)
plt.plot (residual , label = 'Residuals')
plt.legend (loc = 'best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace = True)
test_stationarity(decomposedLogData)
                                                            '''Results of Dickey-Fuller Test:
                                                    Test Statistic                -6.332387e+00
                                                    p-value                        2.885059e-08
                                                    #lags Used                     9.000000e+00
                                                    Number of Observations Used    1.220000e+02
                                                    Critical Value (1%)           -3.485122e+00
                                                    Critical Value (5%)           -2.885538e+00
                                                    Critical Value (10%)          -2.579569e+00
                                                    dtype: float64'''
# Now the time series is stationary
# Residuals are the irregularities present in the data

# checking whether the noise is stationary or not
decomposedLogData = residual
decomposedLogData.dropna(inplace = True)
test_stationarity(decomposedLogData)                                                        
                                                            '''Results of Dickey-Fuller Test:
                                                    Test Statistic                -6.332387e+00
                                                    p-value                        2.885059e-08
                                                    #lags Used                     9.000000e+00
                                                    Number of Observations Used    1.220000e+02
                                                    Critical Value (1%)           -3.485122e+00
                                                    Critical Value (5%)           -2.885538e+00
                                                    Critical Value (10%)          -2.579569e+00
                                                    dtype: float64'''

# Found the value of d... yet to find p[auto regressive lags] and q [moving average]

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf , pacf

lag_acf = acf(datasetLogDiffShifting , nlags = 20)
lag_pacf = pacf(datasetLogDiffShifting , nlags = 20 , method = 'ols') #ols = ordinary least square

#Plot ACF-> To calculate the value of q:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0 , linestyle = '--' , color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--' , color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--' , color = 'gray')
plt.title ('Autocorrelation Function')

# Plot PACF-> To calculate the value of p:
plt.subplot (122)
plt.plot(lag_pacf)
plt.axhline(y=0 , linestyle = '--' , color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--' , color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--' , color = 'gray')
plt.title ('Partial Autocorrelation Function')
plt.tight_layout()
# To find the value of P and Q check at which point the graph reaches 0 value for the first time

from statsmodels.tsa.arima_model import ARIMA

#AR model
model = ARIMA(indexedDataset_logScale , order = (2,1,2)) # p=2 d =1 q = 2
results_AR = model.fit (disp = -1)
plt.plot (datasetLogDiffShifting)
plt.plot (results_AR.fittedvalues , color = 'red')
plt.title ('RSS: %.4f'% sum ((results_AR.fittedvalues-datasetLogDiffShifting["#Passengers"]))**2)
# RSS - Residual Sum of Square
# Greter the RSS the bad it is for us                                                                             
print ('Plotting AR model')                                                                            

# MA model
model = ARIMA (indexedDataset_logScale , order = (2,1,0))
results_MA = model.fit (disp = -1)
plt.plot (datasetLogDiffShifting)
plt.plot (results_MA.fittedvalues , color = 'red')
plt.title ('RSS: 1.5023')
plt.title ('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["#Passengers"]))**2)
print ('Plotting the MA model')
                                                     
model = ARIMA(indexedDataset_logScale , order = (2,1,2))
results_ARIMA = model.fit (disp = -1)
plt.plot (datasetLogDiffShifting)
plt.plot (results_ARIMA.fittedvalues , color = 'red')
plt.title ('RSS: 1.0292')
plt.title ('RSS: %.4f' %sum((results_ARIMA.fittedvalues-datasetLogDiffShifting["#Passengers"]))**2)                                                    
                                                    
# Combining the model to ARIMA
predictions_ARIMA_diff = pd.Series (results_ARIMA.fittedvalues , copy = True)
print (predictions_ARIMA_diff.head())

#Convert to cummulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum)

predictions_ARIMA_log = pd.Series (indexedDataset_logScale['#Passengers'].ix[0] , index = indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add (predictions_ARIMA_diff_cumsum , fill_value = 0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp (predictions_ARIMA_log) # to bring back the original data from log 
plt.plot (indexedDataset)
plt.plot (predictions_ARIMA)
#only the magnitude is varying but the shape has been properly captured

results_ARIMA.plot_predict(1,264) # 264 (data points) = total_rows [144] + number_of_years_for_which_prediction_is_made[10*12]
# if we want to predict for 10 years -> 12[rows]*10 = 120 datapoints
x = results_ARIMA.forecast (steps = 120)

results_ARIMA.forecast (steps = 120)













