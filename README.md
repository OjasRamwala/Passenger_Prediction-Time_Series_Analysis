# Passenger_Prediction-Time_Series_Analysis
This project is a learning attempt to build a model to forecast the demand (passenger traffic) in Airplanes. Time Series Analysis is used to build this model.


Why Time Series Analysis?

In some cases we just have 1 variable - Time.
We analyse the time series data in order to extract meaningful statistics and other characteristics.

What is Time Series? 
A time series is a set of observation taken at specified times usually at equal intervals.
It is used to predict the future values based on the previous observed values.


Applications:-
  
  (1) Business Forecasting
  
  (2) Unterstanding the past behaviour
  
  (3) Planning the future
  
  (4) Evaluate current accomplishments
  
Components of a Time Series:-
  (1) Trend (up-trend or down-trend or horizontal/stationary trend) - Overall long term persistent movement
  (2) Seasonality (Repeating pattern within a fixed time period) - Regular periodic fluctuations
  (3) Irregularity (-Noise , eratic in nature , unsystematic or residual - For short duration and not repeating)
  (4) Cyclic (Repeating up and down movements or swings)
  
Time Series may not be applied in cases when (basically there is no point in applying):-
   (1) Values are constant
   (2) Values are in the form of functions
 
What is Stationarity?
Time Series has a particular behaviour over time.
There is a very high probabilty that it will follow the same in the future.
No matter how much we try there will always be some stationarity in Time Series.
A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time.

Tests to check Stationarity:-
  (1) Rolling Statistics (Visual Sequence) - Plot the moving average or moving variance and see if it varies with time.
  (2) ADCF(Augmented Disckey Fuller)Test -  Null hypothesis is that the TS is non-stationary. The test results comprise of a Test                                                   Statistic and some Critical Values. 
 ARIMA MODEL :- Best model to work with time series data.
 AR + I + MA = Auto Regressive + Integration + Moving Average (combination of 2 models)
 ARIMA model has 3 parameters : P = Autoregressive Lags ; Q = moving average ; d = order of differentiation
 To predict the value of P we will have to plot the Partial Auto Correlation (PAC) graph.
 To predict the value of Q we will have to plot Auto Correlation Function (ACF) graph.
 

   
  
  
  
