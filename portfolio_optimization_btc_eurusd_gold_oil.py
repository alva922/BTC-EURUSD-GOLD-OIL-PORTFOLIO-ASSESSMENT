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
