#Joint Analysis of Bitcoin, Gold and Crude Oil Prices
#https://wp.me/pdMwZd-5gc
#https://medium.com/@alexzap922/risk-adjusted-btc-gold-oil-eurusd-portfolio-optimization-for-quant-traders-autoeda-scipy-slsqp-79d0a891f384?sk=997866af1263dec068894a93acad4a44
import os
os.chdir('YOURPATH')    # Set working directory
os. getcwd() 
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA, ARMA
import statsmodels.tsa.arima_model  
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from math import sqrt 
import seaborn as sns
import pylab
import scipy
import yfinance
from arch import arch_model
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

# to ignore some warnings in python when fitting models ARMA or ARIMA
import warnings 
warnings.filterwarnings('ignore')
#TIME SERIES ANALYSIS OF BITCOIN, GOLD AND CRUDE OIL PRICES
# Import data
data = yfinance.download(tickers= 'BTC-USD, GC=F, CL=F, EURUSD=X', 
                         start="2022-01-03", end="2024-05-04", 
                         group_by='column')['Adj Close']
#data
#!pip install pandas_profiling
#!pip install sweetviz
#!pip install autoviz
#!pip install channels
#!pip install django-channels
#!pip install dabl
import pandas as pd
from pandas_profiling import ProfileReport

profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_file("report2024a.html")
import sweetviz as sv
import pandas as pd


# Create an analysis report for your data
report = sv.analyze(data)

# Display the report
report.show_html()

# Check the number of missing values of the raw data
data.isna().sum()

icker
BTC-USD       0
CL=F        264
EURUSD=X    242
GC=F        264
dtype: int64

#Convert currency from USD to EUR
data['btc'] = data['BTC-USD']/data['EURUSD=X']
data['gold'] = data['GC=F']/data['EURUSD=X']
data['oil'] = data['CL=F']/data['EURUSD=X']

#Create 3 Data Frames without missing values

df1 = data['btc']
df2 = data['gold']
df3 = data['oil']

df1 = df1.to_frame()
df2 = df2.to_frame()
df3 = df3.to_frame()

df1.dropna(axis=0, inplace=True)
df2.dropna(axis=0, inplace=True)
df3.dropna(axis=0, inplace=True)

#Rolling Mean & Standard Deviation

df1['rolling_mean_btc'] = df1.btc.rolling(window=12).mean()
df1['rolling_std_btc'] = df1.btc.rolling(window=12).std()
df1.plot(title='Rolling Mean and Rolling Standard Deviation of Bitcoin Prices',
figsize=(15,6))
plt.ylabel("EUR")
plt.show()

df2['rolling_mean_g'] = df2.gold.rolling(window=12).mean()
df2['rolling_std_g'] = df2.gold.rolling(window=12).std()
df2.plot(title='Rolling Mean and Rolling Standard Deviation of Gold Prices',figsize=(15,5))
plt.ylabel('EUR per ounce')
plt.show()

df3['rolling_mean_o'] = df3.oil.rolling(window=12).mean()
df3['rolling_std_o'] = df3.oil.rolling(window=12).std()
df3.plot(title='Rolling Mean and Rolling Standard Deviation of Crude Oil Prices',
figsize=(15,5))
plt.ylabel('EUR per barrel')
plt.show()

df1['rolling_std_btc'].plot(title='Rolling standard deviation of Bitcoin prices', figsize=(10,3))
plt.show()
df2['rolling_std_g'].plot(title='Rolling standard deviation of Gold Prices', figsize=(10,3))
plt.show()
df3['rolling_std_o'].plot(title='Rolling standard deviation of Crude Oil Prices', figsize=(10,3))
plt.show()

#Statistical Analysis Testing & Fitting

fig, ax = plt.subplots(1, 3, figsize=(20,5))

sgt.plot_acf(df1.btc, lags=40, zero=False, ax=ax[0])
sgt.plot_acf(df2.gold, lags=40, zero=False, ax=ax[1])
sgt.plot_acf(df3.oil, lags=40, zero=False, ax=ax[2])
fig.suptitle('ACF of Bitcoin (left), Gold (center), and Crude Oil (right) Prices', fontsize=16)
plt.show()

#Let's run the ADF test

sts.adfuller(df1.btc)

(-1.5503385145013782,
 0.5084790993259574,
 1,
 823,
 {'1%': -3.438320611647225,
  '5%': -2.8650582305072954,
  '10%': -2.568643405981436},
 13734.333273508622)

sts.adfuller(df2.gold)

(-0.8024242489303032,
 0.8184089254307794,
 2,
 567,
 {'1%': -3.441935806025943,
  '5%': -2.8666509204896093,
  '10%': -2.5694919649816947},
 4665.361428783758)

sts.adfuller(df3.oil)

(-1.7540945328321098,
 0.4034786580961352,
 11,
 559,
 {'1%': -3.442102384299813,
  '5%': -2.8667242618524233,
  '10%': -2.569531046591633},
 2449.1714456088794)

#Let's convert prices to log returns

df1['log_ret_btc'] = np.log(df1.btc/df1.btc.shift(1))

df2['log_ret_g'] = np.log(df2.gold/df2.gold.shift(1))

df3['trans_o'] = df3['oil'] + 1 - df3['oil'].min()
df3['log_ret_o'] = np.log(df3.trans_o/df3.trans_o.shift(1))

#Check the missing values 
print('Number of missing values of Bitcoin log returns:', df1.log_ret_btc.isna().sum())
print('Number of missing values of Gold log returns:', df2.log_ret_g.isna().sum())
print('Number of missing values of Crude oil log returns:', df3.log_ret_o.isna().sum())

Number of missing values of Bitcoin log returns: 1
Number of missing values of Gold log returns: 1
Number of missing values of Crude oil log returns: 1

#Let's plot 3 series of log returns
fig, ax = plt.subplots(3, 1, figsize=(10,9))
df1.log_ret_btc.plot(ax=ax[0])
df2.log_ret_g.plot(ax=ax[1])
df3.log_ret_o.plot(ax=ax[2])
fig.suptitle('Bitcoin (top), Gold (middle), and Crude Oil (bottom) Log Returns', fontsize=16)
plt.show()

#Let's compare the corresponding histograms
fig, ax = plt.subplots(1, 3, figsize=(20,5))
sns.distplot(df1.log_ret_btc[1:], ax=ax[0])
sns.distplot(df2.log_ret_g[1:], ax=ax[1])
sns.distplot(df3.log_ret_o[1:], ax=ax[2])
fig.suptitle('Histograms of Log Returns of Bitcoin (left), Gold (middle), and Crude Oil (right)', fontsize=16)
plt.show()

#Let's check the mean, skewness and kurtosis of log returns
print('- Bitcoin log returns:')
print('Mean:', df1.log_ret_btc.mean())
print('Skewness:', df1.log_ret_btc[1:].skew())
print('Kurtosis:', df1.log_ret_btc[1:].kurtosis())
print('\n')
print('- Gold log returns:')
print('Mean:', df2.log_ret_g.mean())
print('Skewness:', df2.log_ret_g[1:].skew())
print('Kurtosis:', df2.log_ret_g[1:].kurtosis())
print('\n')
print('- Crude oil log returns:', df3.log_ret_o.mean())
print('Skewness:', df3.log_ret_o[1:].skew())
print('Kurtosis:', df3.log_ret_o[1:].kurtosis())

- Bitcoin log returns:
Mean: 7.764716963697599e-05
Skewness: -0.2460688967564316
Kurtosis: 2.771782541387675

- Gold log returns:
Mean: 0.0002631919718611769
Skewness: -0.08269179477267279
Kurtosis: 0.6197039104931932

- Crude oil log returns: 0.006271078632757689
Skewness: 4.001251918715912
Kurtosis: 56.558220392990364

#Let's plot ACFs of log returns without zero lag

fig, ax = plt.subplots(1, 3, figsize=(20,5))

sgt.plot_acf(df1.log_ret_btc[1:], lags=40, zero=False, ax=ax[0])
sgt.plot_acf(df2.log_ret_g[1:], lags=40, zero=False, ax=ax[1])
sgt.plot_acf(df3.log_ret_o[1:], lags=40, zero=False, ax=ax[2])
fig.suptitle('ACF of Bitcoin(left), Gold(center), and Crude Oil(right) Log Returns', fontsize=16)
plt.show()

