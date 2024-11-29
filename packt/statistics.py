# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

#standard variance of data set using std() function
std_dev = df.std()
print(std_dev)
# standard variance of the specific column
sv_height=df.loc[:,"height"].std()
print(sv_height)

# Measures of Variance
# variance of data set using var() function
variance=df.var()
print(variance)
# variance of the specific column
var_height=df.loc[:,"height"].var()
print(var_height)

df.loc[:,"height"].var()
df.skew()
# skewness of the specific column
df.loc[:,"height"].skew()

# Kurtosis of data in data using skew() function
# Kurtosis of the specific column
sk_height=df.loc[:,"height"].kurt()
print(sk_height)

sns.set()
plt.rcParams['figure.figsize'] = (10, 6)
# plot the relationship between “engine-size” and ”price”
plt.scatter(df["price"], df["engine-size"])
plt.title("Scatter Plot for engine-size vs price")
plt.xlabel("engine-size")
plt.ylabel("price")
plt.show()

#boxplot to visualize the distribution of "price" with types of "drive-wheels"
sns.boxplot(x="drive-wheels", y="price",data=df, palette='Set1')
plt.show()
type(df.price[0])

# Calculate percentiles
# calculating 30th percentile of heights in dataset
height = df["height"]
percentile = np.percentile(height, 50,)
print(percentile)

price = df.price.sort_values()
Q1 = np.percentile(price, 25)
Q2 = np.percentile(price, 50)
Q3 = np.percentile(price, 75)

# The IQR is not affected by the presence of outliers
IQR = Q3 - Q1
IQR

df["normalized-losses"].describe()

scorePhysics = [34,35,35,35,35,35,36,36,37,37,37,37,37,38,38,38,39,39,
              40,40,40,40,40,41,42,42,42,42,42,42,42,42,43,43,43,43,44,44,44,44,44,44,45,
              45,45,45,45,46,46,46,46,46,46,47,47,47,47,47,47,48,48,48,48,48,49,49,49,49,
              49,49,49,49,52,52,52,53,53,53,53,53,53,53,53,54,54,
              54,54,54,54,54,55,55,55,55,55,56,56,56,56,56,56,57,57,57,58,58,59,59,59,59,
              59,59,59,60,60,60,60,60,60,60,61,61,61,61,61,62,62,63,63,63,63,63,64,64,64,
              64,64,64,64,65,65,65,66,66,67,67,68,68,68,68,68,68,68,69,70,71,71,71,72,72,
              72,72,73,73,74,75,76,76,76,76,77,77,78,79,79,80,80,81,84,84,85,85,87,87,88]
            
scoreLiterature = [49,49,50,51,51,52,52,52,52,53,54,54,55,55,55,55,56,
                 56,56,56,56,57,57,57,58,58,58,59,59,59,60,60,60,60,60,60,60,61,61,61,62,
                 62,62,62,63,63,67,67,68,68,68,68,68,68,69,69,69,69,69,69,
                 70,71,71,71,71,72,72,72,72,73,73,73,73,74,74,74,74,74,75,75,75,76,76,76,
                 77,77,78,78,78,79,79,79,80,80,82,83,85,88]
                 
scoreComputer = [56,57,58,58,58,60,60,61,61,61,61,61,61,62,62,62,62,
                63,63,63,63,63,64,64,64,64,65,65,66,66,67,67,67,67,67,67,67,68,68,68,69,
                69,70,70,70,71,71,71,73,73,74,75,75,76,76,77,77,77,78,78,81,82,
                84,89,90]

scores=[scorePhysics, scoreLiterature, scoreComputer]
plt.boxplot(scoreComputer, showmeans=True, whis = 99)
plt.show()

box = plt.boxplot(scores, showmeans=True, whis=99)

plt.setp(box['boxes'][0], color='blue')
plt.setp(box['caps'][0], color='blue')
plt.setp(box['caps'][1], color='blue')
plt.setp(box['whiskers'][0], color='blue')
plt.setp(box['whiskers'][1], color='blue')

plt.setp(box['boxes'][1], color='red')
plt.setp(box['caps'][2], color='red')
plt.setp(box['caps'][3], color='red')
plt.setp(box['whiskers'][2], color='red')
plt.setp(box['whiskers'][3], color='red')

plt.ylim([20, 95]) 
plt.grid(True, axis='y')  
plt.title('Distribution of the scores in three subjects', fontsize=18) 
plt.ylabel('Total score in that subject')            
plt.xticks([1,2,3], ['Physics','Literature','Computer'])
plt.show()

df.head()
df.groupby('body-style').groups.keys()
style = df.groupby('body-style')
#To print the values contained in group convertible
style.get_group("convertible")

double_grouping = df.groupby(["body-style","drive-wheels"])
#To print the first values contained in each group
double_grouping.first()

# max() will print the maximum entry of each group 
style['normalized-losses'].max() #output in series
style[['normalized-losses']].max() #output in dataframe
# min() will print the minimum entry of each group 
style['normalized-losses'].min()
style[['normalized-losses']].min()

style.get_group('convertible').mean()
# get the sum of the price for each body-style group
style['price'].sum()
# get the number of symboling/records in each group
style['symboling'].count()