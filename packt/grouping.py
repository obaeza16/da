import pandas as pd
import numpy as np

left1 = pd.DataFrame({'key': ['apple','ball','apple', 'apple', 'ball', 'cat'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [33.4, 5]}, index=['apple', 'ball'])
print(left1)
print(right1)

# Join two dataframes by key
df1 = pd.merge(left1, right1, left_on='key', right_index=True)
df1

# Join two dataframes in outer way
df2 = pd.merge(left1, right1, left_on='key', right_index=True, how='outer')
df2

yarra = np.arange(15).reshape((3,5))
yarra

np.concatenate([yarra,yarra],axis=1)

# new dataframe that consist length,width,height,curb-weight and price
df = pd.read_csv("https://raw.githubusercontent.com/PacktPublishing/hands-on-exploratory-data-analysis-with-python/master/Chapter%205/data.csv")
df.head()

new_dataset = df.filter(["length","width","height","curb-weight","price"],axis=1)
new_dataset.head()
new_dataset.info()
new_dataset.any().isna()

new_dataset['curb-weight'] = new_dataset['curb-weight'].astype(float)

new_dataset['price'].str.isnumeric().value_counts()
# List out the values which are not numeric
new_dataset['price'].loc[new_dataset['price'].str.isnumeric() == False]
#Setting the missing value to mean of price and convert the datatype to integer
price = new_dataset['price'].loc[new_dataset['price'] != '?']
pmean = price.astype(str).astype(int).mean()
new_dataset['price'] = new_dataset['price'].replace('?',pmean).astype(float)
# Check all columns are the needed type
new_dataset.info()

# applying single aggregation for mean over the columns
new_dataset.agg('mean', axis='rows')
# applying aggregation sum and minimun across all the columns 
new_dataset.agg(['sum', 'min'])
# find aggregation for these columns 
new_dataset.aggregate({"length":['sum', 'min'], 
              "width":['max', 'min'], 
              "height":['min', 'sum'],  
              "curb-weight":['sum']}) 
# if any specific aggregation is not applied on a column
# then it has NaN value corresponding to it
df.dtypes

# Group the data frame df by body-style and drive-wheels and extract stats from each group
df.groupby(["body-style","drive-wheels"]).agg({'height': min, 'length': max, 'price': 'mean'})


# create dictionary of aggregrations
aggregations=(
    {
         'height':min,    # minimum height of car in each group
         'length': max,  # maximum length of car in each group
         'price': 'mean',  # average price of car in each group
        
    }
)
# implementing aggregations in groups
df.groupby(
   ["body-style","drive-wheels"]
).agg(aggregations) 

# using numpy libraries for operations
df.groupby(
   ["body-style","drive-wheels"])["price"].agg([np.sum, np.mean, np.std])

# Renaming grouped aggregation columns
df.groupby(
   ["body-style","drive-wheels"]).agg(
    # Get max of the price column for each group
    max_price=('price', max),
    # Get min of the price column for each group
    min_price=('price', min),
    # Get sum of the price column for each group
    total_price=('price', 'mean')     
)
   
# Group wise transformations
df["price"] = df["price"].transform(lambda x:x + x/10)
df.loc[:,'price']

df.groupby(["body-style","drive-wheels"])["price"].transform('mean')
df["average-price"]=df.groupby(["body-style","drive-wheels"])["price"].transform('mean')
# selectiing columns body-style,drive-wheels,price and average-price
df.loc[:,["body-style","drive-wheels","price","average-price"]]


new_dataset1 = df.filter(["body-style","drive-wheels",
                          "length","width","height","curb-weight","price"],axis=1)
new_dataset.dtypes
new_dataset1
#simplest pivot table with dataframe df and index body-style
table = pd.pivot_table(new_dataset1, index =["body-style","drive-wheels"], aggfunc='mean')
table

#new data set with few columns
new_dataset3 = df.filter(["body-style","drive-wheels","price"],axis=1)

# pivot table with dataset new_dataset2
# values are the column in which aggregration function is to be applied
# index is column for grouping of data
# columns for specifying category of data 
# aggfunc is the aggregration function to be applied
# fill_value to fill missing values
table2 = pd.pivot_table(new_dataset3, values='price', index=["body-style"],
                       columns=["drive-wheels"],aggfunc=np.mean,fill_value=0)
table2

