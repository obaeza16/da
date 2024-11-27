import pandas as pd
import numpy as np
from datetime import datetime

dataFrame1 =  pd.DataFrame({ 'StudentID': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
                            'Score' : [89, 39, 50, 97, 22, 66, 31, 51, 71, 91, 56, 32, 52, 73, 92]})
dataFrame2 =  pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                            'Score': [98, 93, 44, 77, 69, 56, 31, 53, 78, 93, 56, 77, 33, 56, 27]})

# In the dataset above, the first column contains information about student identifier
# and the second column contains their respective scores in any subject. The structure
# of the dataframes is same in the both cases. In this case, we would need to concatenate both of them. 

# We can do that by using Pandas concat() method. 

dataframe = pd.concat([dataFrame1, dataFrame2], ignore_index=True)
dataframe

# Merging
df1SE =  pd.DataFrame({ 'StudentID': [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], 'ScoreSE' : [22, 66, 31, 51, 71, 91, 56, 32, 52, 73, 92]})
df2SE =  pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], 'ScoreSE': [98, 93, 44, 77, 69, 56, 31, 53, 78, 93, 56, 77, 33, 56, 27]})

df1ML =  pd.DataFrame({ 'StudentID': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], 'ScoreML' : [39, 49, 55, 77, 52, 86, 41, 77, 73, 51, 86, 82, 92, 23, 49]})
df2ML =  pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 'ScoreML': [93, 44, 78, 97, 87, 89, 39, 43, 88, 78]})

# Option 1
dfSE = pd.concat([df1SE, df2SE], ignore_index=True)
dfML = pd.concat([df1ML, df2ML], ignore_index=True)

df = pd.concat([dfML, dfSE], axis=1)
df

# Option 2
dfSE = pd.concat([df1SE, df2SE], ignore_index=True)
dfML = pd.concat([df1ML, df2ML], ignore_index=True)

df = dfSE.merge(dfML, how='inner')
df

# Here, you will perform inner join with each dataframe. That is to say, if an item exists
# on the both dataframe, will be included in the new dataframe. This means, we will get the
# list of students who are appearing in both the courses. 

# Option 3
dfSE = pd.concat([df1SE, df2SE], ignore_index=True)
dfML = pd.concat([df1ML, df2ML], ignore_index=True)

df = dfSE.merge(dfML, how='left')
df

# Option 4
dfSE = pd.concat([df1SE, df2SE], ignore_index=True)
dfML = pd.concat([df1ML, df2ML], ignore_index=True)

df = dfSE.merge(dfML, how='right')
df

# Option 5
dfSE = pd.concat([df1SE, df2SE], ignore_index=True)
dfML = pd.concat([df1ML, df2ML], ignore_index=True)

df = dfSE.merge(dfML, how='outer')
df

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/PacktPublishing/hands-on-exploratory-data-analysis-with-python/master/Chapter%204/sales.csv')
df.head(10)
df.sort_values(by='Quantity',ascending=False)

#Add new colum that is the total price based on the quantity and the unit price
df['TotalPrice'] = df['UnitPrice'] * df['Quantity']
df.head(10)

df['Company'].value_counts()
df.describe()

# Reshaping with Hierarchical Indexing
data = np.arange(15).reshape((3,5))
indexers = ['Rainfall', 'Humidity', 'Wind']
dframe1 = pd.DataFrame(data, index=indexers, columns=['Bergen', 'Oslo', 'Trondheim', 'Stavanger', 'Kristiansand'])
dframe1

stacked = dframe1.stack()
print(stacked)
stacked.unstack()

series1 = pd.Series([000, 111, 222, 333], index=['zeros','ones', 'twos', 'threes'])
series2 = pd.Series([444, 555, 666], index=['fours', 'fives', 'sixs'])

frame2 = pd.concat([series1, series2], keys=['Number1', 'Number2'])
frame2.unstack()
frame2

# Data Deduplication
frame3 = pd.DataFrame({'column 1': ['Looping'] * 3 + ['Functions'] * 4, 'column 2': [10, 10, 22, 23, 23, 24, 24]})
frame3

frame3.duplicated()

frame4 = frame3.drop_duplicates()
frame4

frame3['column 3'] = range(7)
frame3
frame5 = frame3.drop_duplicates(['column 2'])
frame5

# Replacing values
replaceFrame = pd.DataFrame({'column 1': [200., 3000., -786., 3000., 234., 444., -786., 332., 3332. ], 'column 2': range(9)})
replaceFrame
replaceFrame.replace(to_replace =-786, value= np.nan)

replaceFrame = pd.DataFrame({'column 1': [200., 3000., -786., 3000., 234., 444., -786., 332., 3332. ], 'column 2': range(9)})
replaceFrame
replaceFrame.replace(to_replace =[-786, 0], value= [np.nan, 2])

# Handling missing data
data = np.arange(15, 30).reshape(5, 3)
dfx = pd.DataFrame(data, index=['apple', 'banana', 'kiwi', 'grapes', 'mango'], columns=['store1', 'store2', 'store3'])
dfx

dfx['store4'] = np.nan
dfx.loc['watermelon'] = np.arange(15, 19)
dfx.loc['oranges'] = np.nan
dfx['store5'] = np.nan
dfx['store4']['apple'] = 20.
dfx

dfx.isna()
dfx.notna()

dfx.isna().sum()
dfx.isna().sum().sum()
dfx.count()
dfx.store4[dfx.store4.notnull()]
dfx.store4.dropna()
dfx.dropna()
dfx.dropna(how='all')
dfx.dropna(how='all', axis=1)

dfx2 = dfx.copy()
dfx2.loc['oranges'].store1 = 0
dfx2.loc['oranges'].store3 = 0
dfx2
dfx2.dropna(how='any', axis=1)
dfx.dropna(thresh=5, axis=1)

# NaN in mathematical applications
ar1 = np.array([100, 200, np.nan, 300])
ser1 = pd.Series(ar1)

ar1.mean(), ser1.mean()
ser2 = dfx.store4
ser2.sum()
ser2.mean()
ser2.cumsum()

dfx2.store4 + 1

# Filling in missing data
filledDf = dfx.fillna(0)
filledDf

dfx.mean()
filledDf.mean()

# Forward and backward filling of the missing values
dfx2.store4.ffill()
dfx2.store4.bfill()

# Filling with index labels
to_fill = pd.Series([14, 23, 12], index=['apple', 'mango', 'oranges'])
to_fill
dfx.store4.fillna(to_fill)
dfx.fillna(dfx.mean())

# Interpolation of missing values
ser3 = pd.Series([100, np.nan, np.nan, np.nan, 292])
ser3
ser3.interpolate()

ts = pd.Series([10, np.nan, np.nan, 9], 
               index=[datetime(2019, 1,1), 
                      datetime(2019, 2,1), 
                      datetime(2019, 3,1),
                      datetime(2019, 5,1)])

ts
ts.interpolate()
ts.interpolate(method='time')

# Renaming axis indices
data = np.arange(15).reshape((3,5))
indexers = ['Rainfall', 'Humidity', 'Wind']
dframe1 = pd.DataFrame(data, index=indexers, columns=['Bergen', 'Oslo', 'Trondheim', 'Stavanger', 'Kristiansand'])
dframe1

# Say, you want to transform the index terms to capital letter. 
dframe1.index = dframe1.index.map(str.upper)
dframe1

dframe1.rename(index=str.title, columns=str.upper)

# Discretization and binning
height =  [120, 122, 125, 127, 121, 123, 137, 131, 161, 145, 141, 132]

bins = [118, 125, 135, 160, 200]

category = pd.cut(height, bins)
category

pd.Series(category).value_counts()
pd.value_counts(category)

category2 = pd.cut(height, [118, 126, 136, 161, 200], right=False)
category2

bin_names = ['Short Height', 'Averge height', 'Good Height', 'Taller']
pd.cut(height, bins, labels=bin_names)
pd.cut(np.random.rand(40), 5, precision=2)

randomNumbers = np.random.rand(2000)
category3 = pd.qcut(randomNumbers, 4) # cut into quartiles
category3

pd.Series(category3).value_counts()

pd.qcut(randomNumbers, [0, 0.3, 0.5, 0.7, 1.0])

# Load data from github
df = pd.read_csv('https://raw.githubusercontent.com/PacktPublishing/hands-on-exploratory-data-analysis-with-python/master/Chapter%204/sales.csv')
df.head(10)
df.info(memory_usage='deep')
df.describe()

# Find values in order that exceeded 
df['TotalPrice'] = df['UnitPrice'] * df['Quantity']
df.head(10)
# Find transaction exceeded 3000000
TotalTransaction = df["TotalPrice"]
TotalTransaction[np.abs(TotalTransaction) > 3000000]
df[np.abs(TotalTransaction) > 6741112]

# Permutation and random sampling
dat = np.arange(80).reshape(10,8)
df = pd.DataFrame(dat)
df

sampler = np.random.permutation(10)
sampler
df.take(sampler)

# Random sample without replacement
df.take(np.random.permutation(len(df))[:3])
# Random sample with replacement
sack = np.array([4, 8, -2, 7, 5])
sampler = np.random.randint(0, len(sack), size = 10)
sampler

draw = sack.take(sampler)
draw

# Dummy variables
df = pd.DataFrame({'gender': ['female', 'female', 'male', 'unknown', 'male', 'female'], 'votes': range(6, 12, 1)})
df

pd.get_dummies(df['gender'], dtype=float)
dummies = pd.get_dummies(df['gender'], prefix='gender')
dummies
with_dummy = df[['votes']].join(dummies)
with_dummy
