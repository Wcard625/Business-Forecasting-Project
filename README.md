# Business-Forecasting-Project
Harmonized Index of Consumer Prices (HICP) - specifically in Norway for products such as sugar, jam, honey, chocolate, and other confectionary goods


This project is to forecast the HICP value for the basket goods mentioned above in Norway by processing the data from 2010-2021. The index tracks the prices of goods such as coffee, tobacco, meat, fruit, household appliances, cars, pharmaceuticals, electricity, clothing and many other widely used products. Furthermore, the metric is also used as a meausre of inflation in the European Union. Forecasting HICP can be used to understand the inflation rate and how to protect against increases or decreases in the prices of common consumer goods which can have significant application in cost savings for a business in the area. 

The project forecast could be a Point forecast along with a confidence interval for HICP in the next year (2022).

For this forecast, we will be utilizing the following accuracy measures: MAD Mean Absolute Value , MSE Mean Squared Error, RMSE Root Mean Squared Error, MAPE Mean Absolute Percentage Error, MPE Mean Percentage Error . We are leaning more towards using RMSE as an accuracy measure as we would like an accuracy measure that would penalise us for errors in this particular prediction.

The data is coming straight from the Federal Reserve Economic Database - FRED. (https://fred.stlouisfed.org/series/CP0118NOM086NEST) The frequency of the data is monthly starting from 2010 till 2021 August. The Harmonized Index of Consumer Prices category "Sugar, Jam, Honey, Chocolate, and Confectionery (01.1.8)" is a classification of nondurable goods that includes cane or beet sugar, unrefined or refined sugar, powdered sugar, crystallized sugar, or sugar lumps; jams, marmalades, chocolates etc.
