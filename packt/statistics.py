# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, Latex
from IPython.core.display import Image
import seaborn as sns
# Uniform Distribution
from scipy.stats import uniform
# Normal distribution
from scipy.stats import norm
# Gamma distribution
from scipy.stats import gamma
# Exponential distribution
from scipy.stats import expon
# Poisson Distribution
from scipy.stats import poisson
# Binomial Distribution
from scipy.stats import binom

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(10,6)})

number = 10000
start = 20
width = 25

uniform_data = uniform.rvs(size=number, loc=start, scale=width)
# Deprecated!
# axis = sns.distplot(uniform_data, bins=100, kde=True, color='skyblue', hist_kws={"linewidth": 15})
# axis.set(xlabel='Uniform Distribution ', ylabel='Frequency')
# plt.show()

axis2 = sns.histplot(uniform_data,bins=100, kde=True, color='skyblue', kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
axis2.set(xlabel='Uniform Distribution ', ylabel='Frequency')
plt.show()

normal_data = norm.rvs(size=90000,loc=20,scale=30)
axis = sns.histplot(normal_data, bins=100, kde=True, color='skyblue', kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
axis.set(xlabel='Normal Distribution', ylabel='Frequency')
plt.show()

gamma_data = gamma.rvs(a=5, size=10000)
axis = sns.histplot(gamma_data, kde=True, bins=100, color='skyblue', kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
axis.set(xlabel='Example of Gamma Distribution', ylabel='Frequency')
plt.show()

expon_data = expon.rvs(scale=1,loc=0,size=1000)
axis = sns.histplot(expon_data, kde=True, bins=100, color='skyblue', kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
axis.set(xlabel='Exponential Distribution', ylabel='Frequency')
plt.show()

poisson_data = poisson.rvs(mu=2, size=10000)
axis = sns.histplot(poisson_data, bins=30, kde=False, color='red', kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
axis.set(xlabel='Poisson Distribution', ylabel='Frequency')
plt.show()

binomial_data = binom.rvs(n=10, p=0.8,size=10000)
axis = sns.histplot(binomial_data, kde=False, color='red', kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
axis.set(xlabel='Binomial Distribution', ylabel='Frequency')
plt.show()

# loading data set as Pandas dataframe
df = pd.read_csv("https://raw.githubusercontent.com/PacktPublishing/hands-on-exploratory-data-analysis-with-python/master/Chapter%205/data.csv")
df.head()

df.dtypes

# Data Cleaning
# Find out the number of values which are not numeric
df['price'].str.isnumeric().value_counts()

# List out the values which are not numeric
df['price'].loc[df['price'].str.isnumeric() == False]
#Setting the missing value to mean of price and convert the datatype to integer
price = df['price'].loc[df['price'] != '?']
pmean = price.astype(str).astype(int).mean()
df['price'] = df['price'].replace('?',pmean).astype(int)
df['price'].head()

# Cleaning the horsepower losses field
df['horsepower'].str.isnumeric().value_counts()
horsepower = df['horsepower'].loc[df['horsepower'] != '?']
hpmean = horsepower.astype(str).astype(int).mean()
df['horsepower'] = df['horsepower'].replace('?',hpmean).astype(int)
df['horsepower'].head()

# Cleaning the Normalized losses field
df[df['normalized-losses']=='?'].count()
nl=df['normalized-losses'].loc[df['normalized-losses'] !='?'].count()
nmean=nl.astype(str).astype(int).mean()
df['normalized-losses'] = df['normalized-losses'].replace('?',nmean).astype(int)
df['normalized-losses'].head()

# cleaning the bore
# Find out the number of invalid value
df['bore'].loc[df['bore'] == '?']
# Replace the non-numeric value to null and convert the datatype
df['bore'] = pd.to_numeric(df['bore'],errors='coerce')
df.bore.head()

# Cleaning the column stoke
df['stroke'] = pd.to_numeric(df['stroke'],errors='coerce')
df['stroke'].head()

# Cleaning the column peak-rpm 
df['peak-rpm'].iloc[130]
df['peak-rpm'] = pd.to_numeric(df['peak-rpm'],errors='coerce')
df['peak-rpm'].head()

df['peak-rpm'].value_counts()
df['peak-rpm'].isna().any()

# Cleaning the Column num-of-doors data
# remove the records which are having the value '?'
df['num-of-doors'].loc[df['num-of-doors'] == '?']
df= df[df['num-of-doors'] != '?']
df['num-of-doors'].loc[df['num-of-doors'] == '?']

df.describe()

# Get column heigth from df
height = df['height']
print(height)

# Calculate mean, median and mode of heigth data
mean = height.mean()
median = height.median()
mode = height.mode()
print(mean, median, mode)

df.make.value_counts().nlargest(30).plot(kind='bar')
plt.title('Number of cars by make')
plt.ylabel('Number of cars')
plt.xlabel('Make of the cars')
plt.show()

# summarize categories of drive-wheels
drive_wheels_count = df['drive-wheels'].value_counts()
print(drive_wheels_count)