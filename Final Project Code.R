title: "HICP Forecasting for Norway (2010-2021)"
---
  
  Loading the data for HICP of basket goods (sugar, honey, jam and other confectionary)in Norway 2010-2021

```{r warning=FALSE, message=FALSE}
library(fpp)
library(fpp2)
library(TTR)
HICP_Norway <- read.csv("C:/Users/Downloads/HICP_Norway.csv")
```

```{r}
class(HICP_Norway)
```

Converting the dataset into a time series for forecasting 

```{r warning=FALSE, message=FALSE, fig.width=10,fig.height=7}
hicp_ts <- ts (HICP_Norway$HICP, frequency =12, start=c(2010,1))
hicp_ts
plot (hicp_ts, main="HICP for Basket Goods (Sugar,Honey,Jam,Confectionery) in Norway(2010-2021)",ylab="HICP")
```
Summary of the time series data :
  
  ```{r warning=FALSE, message=FALSE}
summary(hicp_ts)
boxplot(hicp_ts,main="Box-Plot of HICP")
```
## Data Exploration and Preparation
We can see from the time series graph that around the year 2013, there was a dip and post this, we can see a steady increase in the indexes and gradually settling towards the end of 2020 and 2021. This could be due to a fundamental change in the way these goods are priced and thus, for the sake of considering most relevant data for forecasting, we will include the data from 2014 onwards as that would yield a more accurate forecast.

```{r warning=FALSE, message=FALSE,fig.width=10,fig.height=7}
hicp_ts_recent <- window(hicp_ts,start=c(2014,01))
plot(hicp_ts_recent,main="HICP for Norway (2014-2021)",ylab="HICP")
```

We have now cut off the part of data that is no longer relevant to our forecast. We can see the changes in our mean and median after cutting down the historical data.

```{r warning=FALSE, message=FALSE}
summary(hicp_ts_recent)
boxplot(hicp_ts_recent,main="BoxPlot of HICP (2014 onwards)")
```
Comparing the ACF of both time series, we can now see how the seasonality seems to be more pronounced in the new reduced dataset as compared to that in the older dataset.

```{r warning=FALSE, message=FALSE}
Acf(hicp_ts)
Acf(hicp_ts_recent, main="ACF of Time Series")
```

## Decomposition

Decomposition helps us understand if the data has a significant seasonality component. If yes, we can see how the seasonality affects the data.

```{r warning=FALSE, message=FALSE}
decomp_hicp <- decompose(hicp_ts_recent)
plot(decomp_hicp)
```

We can see here that the seasonality component is minuscule. The noise factor is more pronounced than the seasonality component and thus we can say that the data is not affected much by seasonality.
But if we look at the data in general, we can see the index value dips a little every December. This could be attributed to the festive season and how the prices of sugar, jam, honey and other confectionery items are directly affected by it.

To drive this point home, lets take a look at the monthly indices to see how seasonality is pulling the HICP up or down.

```{r warning=FALSE, message=FALSE} 
decomp_hicp$type
round(decomp_hicp$figure/10,2)
```
As we speculated, the seasonality is pretty much insignificant except in Dec where we can see it pulling down the HICP.

Lets try seasonally adjusting the data to see how different it could be from the data at hand.

```{r warning=FALSE, message=FALSE}
hicp_seas_adj <- seasadj(decomp_hicp)
plot(hicp_ts_recent, main="Seasonally Adjusted HICP Data")
lines(hicp_seas_adj,col="Red")
``` 


## Forecasting

### Mean Forecast
We start with the simplest of forecast first which is mean forecast that simply takes the average of the observations and forecasts the value 

```{r warning=FALSE, message=FALSE}
mean_forecast <- meanf(hicp_ts_recent,5)
plot (mean_forecast)
mean_forecast$mean
```

### Naive Forecast
Naive Forecasting principle considers the latest data points to be most relevant. It takes the last data point and predicts the same value over the forecasting period. Obviously, this wouldn't work well in case of data that have seasonality. But it can be taken as a benchmark to compare other model's performance.

```{r warning=FALSE, message=FALSE}
hicp_naive <- naive(hicp_ts_recent,12)
plot(hicp_naive, main='Forecasts from Naive Method (97.8)')
lines(hicp_naive$mean,col="yellow")
```
#### Residual Analysis for Naive

```{r warning=FALSE, message=FALSE}
plot(hicp_naive$residuals,main="Residual Analysis - Naive",ylab="Residuals")
```
ACF of Naive model residuals shows some significant correlation in residuals at Lag 12 and 24. This is expected as we know that there is a seasonal dip happening in Dec every year which the Naive model is not taking into account.

```{r warning=FALSE, message=FALSE}
Acf(hicp_naive$residuals,main="ACF of Naive Residuals")
```

Residual Vs Actual Plot
```{r warning=FALSE, message=FALSE}
plot(hicp_naive$x,hicp_naive$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Actual Values",ylab="Residuals", main="Residual Vs Actual")
```

Residual Vs Fitted Plot
```{r warning=FALSE, message=FALSE}
plot(hicp_naive$fitted,hicp_naive$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Fitted Values",ylab="Residuals",main="Residual Vs Fitted")
```
Residual Histogram Plot

We can see that the residuals show a fairly normal distribution slightly skewed to the right. So the naive method seems to be underforecasting.
```{r warning=FALSE, message=FALSE}
hist(hicp_naive$residuals, main="Histogram of Naive Residuals",xlab="Residuals")
```
```{r warning=FALSE, message=FALSE}
forecast(hicp_naive)
```

#### Accuracy Measures
For this particular project, we would focus on the RMSE as an accuracy measure. This is because forecasting HICP requires high precision as many key economical decisions are made based on the trend seen in HICP. 
As RMSE would penalise us for any error in prediction (since it is squaring up the error), we think this is the best accuracy measure for this particular project.

```{r}
a_n<-accuracy(hicp_naive)
a_n
```
As we can see, the RMSE for Naive is not that great as there is much more information that needs to be utilised by the model to give out better prediction.

### Seasonal Naive Forecast
Seasonal Naive takes the effects of seasonality into account while forecasting and is an improvement over Naive method when forecasting with seasonal data.

```{r warning=FALSE, message=FALSE}
snaive_forecast <- snaive(hicp_ts_recent,12)
plot(snaive_forecast)
lines(snaive_forecast$mean,col="pink")
lines(snaive_forecast$fitted,col="purple")
```

We can see that seasonal naive is not that great at forecasting either as the error in forecasting seems quite huge at places.

### Exponential Smoothing

Exponential smoothing breaks down the data into its components such as seasonality, trend and noise.
```{r warning=FALSE, message=FALSE}
hicp_ets <- ets(hicp_ts_recent)
summary(hicp_ets)
plot(hicp_ets)
```

From the summary we can see that the data shows additive errors, additive trend and additive seasonality.

Mean Squared Error for ets model can be given as :
  
  ```{r warning=FALSE, message=FALSE}
hicp_ets$mse
```

Forecasting can be done by :
  ```{r warning=FALSE, message=FALSE}
forecast_ets <- forecast.ets(hicp_ets, h=12)
forecast_ets
plot(forecast_ets)
```

#### Residual Analysis of ETS 

Histogram of Residuals 
```{r warning=FALSE, message=FALSE}
hist(forecast_ets$residuals, main="Histogram of Residuals for ETS",xlab="Residuals")
```
Similar to Naive, we can see that the residuals are fairly normally ditributed. But as opposed to Naive, we are seeing the histogram skewed to the left meaning our ETS is overforecasting the HICP.

Plot and ACF of Residuals 

```{r warning=FALSE, message=FALSE}
plot(hicp_ets$residuals,main="Residual Analysis - ETS",ylab="Residuals")
Acf(hicp_ets$residuals,main="ACF of ETS Residuals")
```
ACF looks better than the Naive forecast but we still see a couple of significant lags at 12 and 24 for same reasons as stated earlier.

Residuals Vs Actual

The residual plot against both fitted and actual values seem random in nature with no recognizable patterns. 

```{r warning=FALSE, message=FALSE}
plot(hicp_ets$x,hicp_ets$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Actual Values",ylab="Residuals", main="Residual Vs Actual")
```
Residuals Vs Fitted
```{r warning=FALSE, message=FALSE}
plot(hicp_ets$fitted,hicp_ets$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Fitted Values",ylab="Residuals", main="Residual Vs Fitted")
``` 
#### Accuracy Measures
Here are the various accuracy measures for the ETS model. RMSE for ETS seems much better than that of Naive but we can try doing better.


```{r warning=FALSE, message=FALSE}
a_ets<-accuracy(hicp_ets)
a_ets
```


### Holt-Winters Forecast

Holt Winters forecast employs various smoothing constants for level, trend and seasonality components and generates a final forecast equation.Since it takes into consideration all the factors in a time series, we can expect a much better prediction.

```{r warning=FALSE, message=FALSE}
hicp_hw <- HoltWinters(hicp_ts_recent)
hicp_hw
plot(hicp_hw,main="Holt-Winters Fitting")
```
As we can see, the model (red line) perfectly traces the actual time series line (black) including the seasonal dips.

Forecasting using Holt-Winters :
  ```{r warning=FALSE, message=FALSE}
forecast_hw <- forecast(hicp_hw,h=12)
plot(forecast_hw)
forecast_hw$mean
```
We can see Holt Winters predicting further decrease in the HICP value over the coming year.

#### Residual Analysis for Holt-Winters

Residual Plot 

```{r warning=FALSE, message=FALSE}
plot(forecast_hw$residuals,main="Residual Plot for Holt-Winters",ylab="Residuals")
```


ACF of Residuals 

The ACF looks much better than previous Naive or ETS. We can see there are no significant lags in the ACF anymore and thus we can be sure that the errors are not correlated.

```{r warning=FALSE, message=FALSE}
Acf(forecast_hw$residuals, main="ACF of Residuals for Holt-Winters")
```


Histogram of Residuals:
  
  Errors are fairly normally distributed around 0.

```{r warning=FALSE, message=FALSE}
hist(forecast_hw$residuals, main="Histogram of Holt-winters Residuals",xlab="Residuals")
```


Residuals Vs Fitted

```{r warning=FALSE, message=FALSE}
plot(forecast_hw$fitted,forecast_hw$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Fitted Values",ylab="Residuals", main="Residual Vs Fitted")
```

Residuals Vs Actual

```{r warning=FALSE, message=FALSE}
plot(forecast_hw$x,forecast_hw$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Actual Values",ylab="Residuals", main="Residual Vs Actual")
```


#### Accuracy Measures

We can see that the RMSE value for Holt-Winters model is better than Naive but slightly worse than ETS, but the ACF of residuals in case of ETS showed some significant correlation. 
So we would recommend Holt-Winters model over ETS for our forecasting, even though the RMSE measure for ETS is slightly better.

```{r}
a_hw<-accuracy(forecast_hw)
a_hw
sqrt(mean(hicp_hw$SSE))
```

### Simple Moving Averages

Simple moving average is a simple analysis tool that smooths out the data by creating a constantly updated average price.The degree of smoothing depends on the order of the averages being taken.

For order=3,6 and 9
```{r warning=FALSE, message=FALSE}
MA3_hicp <- ma(hicp_ts_recent,order=3)
MA6_hicp <- ma(hicp_ts_recent,order=6)
MA9_hicp <- ma(hicp_ts_recent,order=9)
plot(hicp_ts_recent,ylab="HICP Norway",main="Time Series Plot for HICP - Norway",col="Grey",lwd=1.5)
lines(MA3_hicp, col = "Dark Red",lwd=1.5)
lines(MA6_hicp, col = "Orange",lwd=1.5)
lines(MA9_hicp, col = "Purple",lwd=1.5)
```
Looking at the graph, we believe that moving averages with order 6 would be the best choice for prediction here as we are doing a long range prediction into the next year. 

#### Forecast Using MA

```{r warning=FALSE, message=FALSE}
forecast_ma3 <- forecast(MA3_hicp,12)
forecast_ma6 <- forecast(MA6_hicp,12)
forecast_ma9 <- forecast(MA9_hicp,12)
plot(forecast_ma3)
plot(forecast_ma6)
plot(forecast_ma9)
forecast_ma6$mean
```
We can see from the forecast that the Order 6 moving averages show similar ETS model of Additive error and additive Trend but the additive Seasonality seems to have been smoothened out by the moving averages and thus we dont see any seasonality factor here.

#### Residual Analysis for Simple Moving Averages

We consider the the model with an order of 6 for our forecast as it is a forecast for the next year. 

Residual Plot and Histogram of Residuals :
  
  ```{r warning=FALSE, message=FALSE}
plot(forecast_ma6$residuals,main="Residual Plot for Simple MA",ylab="Residuals")
hist(forecast_ma6$residuals,main="Histogram of Residuals for Simple MA (order=6)",xlab="Residuals")
```

We can see that the residuals are random in nature and are fairly normally distributed around 0 while being slightly skewed to right indicating there may be some underforecasting.

```{r}
Acf(forecast_ma6$residuals)
```
We can see that the ACF of residuals for moving averages are also not satisfactory due to significant lags at 6, 12 and 24.

Residual Vs Actual 

```{r warning=FALSE, message=FALSE}
plot(forecast_ma6$x,forecast_ma6$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Actual Values",ylab="Residuals", main="Residual Vs Actual")
```

Residual Vs Fitted 

```{r warning=FALSE, message=FALSE}
plot(forecast_ma6$fitted,forecast_ma6$residuals,xy.labels=FALSE,xy.lines = FALSE,xlab="Fitted Values",ylab="Residuals", main="Residual Vs Fitted")
```
#### Accuracy Measures for Moving Averages 

Let us take a look at the order 6 moving average model to see the accuracy measures.

```{r}
a_ma <- accuracy(forecast_ma6)
a_ma
```
RMSE value for Order 6 moving averages model seems to be the best one so far but since the ACF od residuals was not satisfactory, we shall move on.

### ARIMA 

Typically used for short range predictions due to its flexibility, we shall now try to do a forecast using ARIMA. 

First, we need to make our data stationary

```{r}
autoplot(hicp_ts_recent,main="Non-Stationary TS plot",ylab="HICP")
```


```{r warning=FALSE, message=FALSE}
adf.test(hicp_ts_recent,k=12)
```
Lets see how it looks after first round of differences

```{r warning=FALSE, message=FALSE}
ndiffs(hicp_ts_recent)
```
We now know that the d part in ARIMA(p,d,q) is 1. 

```{r warning=FALSE, message=FALSE}
nsdiffs(hicp_ts_recent) ##seasonal difference
hicp_d1 <- diff(hicp_ts_recent,differences=1)
tsdisplay(hicp_d1, main="HICP Time Series(Stationary)")
autoplot(hicp_d1,main="Stationary Data",ylab="HICP")
```


Fitting the best ARIMA model

```{r}
auto_fit <- auto.arima(hicp_ts_recent,trace=TRUE,stepwise=FALSE,approximation=FALSE)
auto_fit
```

#### Forecasting Using ARIMA

```{r warning=FALSE, message=FALSE}
arima_hicp <- forecast(auto_fit, h=12)
plot(arima_hicp)
arima_hicp$mean
```
ARIMA model shows that for non-seasonal data requires 1 difference (as d=1) to make the data stationary and moving averages MA(1) as q value is 1. For the seasonal portion of the data, D=0 so there will be no differences required and it would need Auto Regression AR(1) and AR(2) as P=2 for seasonal portion.
It also shows that there is a significant lag at 12.

#### Residual Analysis of ARIMA

```{r}
Acf(auto_fit$residuals,main="ACF of Residuals")
plot.ts(residuals(auto_fit),ylab="Residuals",main="Residula Plot for ARIMA")
hist(auto_fit$residuals, main="Histogram of Residuals (ARIMA)",xlab="Residuals")
```
We can make below observations from residual analysis : 
- There is no significant correlation between the residuals 
- The residuals are random in nature and do not show any recognizable pattern 
- The residuals are normally distributed around 0 while being slightly skewed towards right indicating we may be underforecasting the values.

#### Accuracy Measures

```{r}
a_arma <- accuracy(arima_hicp)
a_arma
```
The accuracy for ARIMA is similar to that of Holt Winters and the ACF shows no significant lags so there is no correlation in residuals. 
ARIMA looks like a good model for our prediction.

### Conclusion

- In conclusion, we can see that Moving Averages give the best RMSE measures. But at the same time the ACF of residuals for Moving Averages was not satisfactory. But looking at Holt-Winters and ARIMA, we can see that RMSE measures as well as ACF of residuals are both satisfactory.
So we should go with one of these models or a combination of both for this particular prediction.

- Going back to the FRED website we could see that HICP for September and October has been updated : 97.6 for September and 96.2 for October. If we look at ARIMA model, we can see that it has made a pretty spot on prediction for Sept 2021 and Oct 2021. This means the ARIMA model is very much reliable. 
