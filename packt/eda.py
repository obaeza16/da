# conda install faker 
# pip install radar

import datetime
import math
import pandas as pd
import random 
import radar
import matplotlib.pyplot as plt
import numpy as np
import calendar
import seaborn as sns
from faker import Faker
fake = Faker()

def generateData(n):
  listdata = []
  start = datetime.datetime(2019, 8, 1)
  end = datetime.datetime(2019, 8, 30)
  delta = end - start
  for _ in range(n):
    date = radar.random_datetime(start='2019-08-1', stop='2019-08-30').strftime("%Y-%m-%d")
    price = round(random.uniform(900, 1000), 4)
    listdata.append([date, price])
  df = pd.DataFrame(listdata, columns = ['Date', 'Price'])
  df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
  df = df.groupby(by='Date').mean()

  return df

df = generateData(50)
df.head(10)

df.to_csv(r'./packt/stock.csv')
plt.plot(df)
plt.show()

# Step 1: Set up the data. Remember range stoping parameter is exclusive. Meaning if you generate range from (1, 13), the last item 13 is not included. 
months = list(range(1, 13))
sold_quantity = [round(random.uniform(100, 200)) for x in range(1, 13)]

# Step 2: Specify the layout of the figure and allocate space. 
figure, axis = plt.subplots()
# Step 3: In the X-axis, we would like to display the name of the months. 
plt.xticks(months, calendar.month_name[1:13], rotation=20)
# Step 4: Plot the graph
plot = axis.bar(months, sold_quantity)
# Step 5: This step can be optinal depending upon if you are interested in displaying the data vaue on the head of the bar. 
# It visually gives more meaning to show actual number of sold iteams on the bar itself. 
for rectangle in plot:
  height = rectangle.get_height()
  axis.text(rectangle.get_x() + rectangle.get_width() /2., 1.002 * height, '%d' % int(height), ha='center', va = 'bottom')

# Step 6: Display the graph on the screen. 
plt.show()

# Scatter Plot
age = list(range(0, 65))
sleep = []

classBless = ['newborns(0-3)', 'infants(4-11)', 'toddlers(12-24)', 'preschoolers(36-60)', 'school-aged-children(72-156)', 'teenagers(168-204)', 'young-adults(216-300)','adults(312-768)', 'older-adults(>=780)']
headers_cols = ['age','min_recommended', 'max_recommended', 'may_be_appropriate_min', 'may_be_appropriate_max', 'min_not_recommended', 'max_not_recommended'] 

# Newborn (0-3)
for i in range(0, 4):
  min_recommended = 14
  max_recommended = 17
  may_be_appropriate_min = 11
  may_be_appropriate_max = 13
  min_not_recommended = 11
  max_not_recommended = 19
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])
  

# infants(4-11)
for i in range(4, 12):
  min_recommended = 12
  max_recommended = 15
  may_be_appropriate_min = 10
  may_be_appropriate_max = 11
  min_not_recommended = 10
  max_not_recommended = 18
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# toddlers(12-24)
for i in range(12, 25):
  min_recommended = 11
  max_recommended = 14
  may_be_appropriate_min = 9
  may_be_appropriate_max = 10
  min_not_recommended = 9
  max_not_recommended = 16
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# preschoolers(36-60)
for i in range(36, 61):
  min_recommended = 10
  max_recommended = 13
  may_be_appropriate_min = 8
  may_be_appropriate_max = 9
  min_not_recommended = 8
  max_not_recommended = 14
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# school-aged-children(72-156)
for i in range(72, 157):
  min_recommended = 9
  max_recommended = 11
  may_be_appropriate_min = 7
  may_be_appropriate_max = 8
  min_not_recommended = 7
  max_not_recommended = 12
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# teenagers(168-204)
for i in range(168, 204):
  min_recommended = 8
  max_recommended = 10
  may_be_appropriate_min = 7
  may_be_appropriate_max = 11
  min_not_recommended = 7
  max_not_recommended = 11
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# young-adults(216-300) 
for i in range(216, 301):
  min_recommended = 7
  max_recommended = 9
  may_be_appropriate_min = 6
  may_be_appropriate_max = 11
  min_not_recommended = 6
  max_not_recommended = 11
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# adults(312-768) 
for i in range(312, 769):
  min_recommended = 7
  max_recommended = 9
  may_be_appropriate_min = 6
  may_be_appropriate_max = 10
  min_not_recommended = 6
  max_not_recommended = 10
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# older-adults(>=780)
for i in range(769, 780):
  min_recommended = 7
  max_recommended = 8
  may_be_appropriate_min = 5
  may_be_appropriate_max = 6
  min_not_recommended = 5
  max_not_recommended = 9
  sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

sleepDf = pd.DataFrame(sleep, columns=headers_cols)
sleepDf.head(10)
sleepDf.to_csv(r'./packt/sleep_vs_age.csv')

sns.set_theme()