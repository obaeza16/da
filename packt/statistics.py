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