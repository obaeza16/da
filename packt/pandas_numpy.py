import numpy as np

# We use try to catch errors:
try:
  print(y)
except NameError:
  print("Well, the variable y is not defined")
except:
  print("OMG, Something else went wrong")
  
y=20

# Without try, if we input not a number it will fail
try:
    Value = int(input("Type a number between 1 and 10:"))
except ValueError:
    print("You must type a number between 1 and 10!")
else:
    if (Value > 0) and (Value <= 10):
        print("You typed: ", Value)
    else:
        print("The value you typed is incorrect!")
        
# Now we start with numpy

# Defining 1D array
my1DArray = np.array([1, 8, 27, 64])
print(my1DArray)

# Defining and printing 2D array
my2DArray = np.array([[1, 2, 3, 4], [2, 4, 9, 16], [4, 8, 18, 32]])
print(my2DArray)

#Defining and printing 3D array
my3Darray = np.array([[[ 1, 2 , 3 , 4],[ 5 , 6 , 7  ,8]], [[ 1,  2,  3,  4],[ 9, 10, 11, 12]]])
print(my3Darray)

# Print out memory address
print(my2DArray.data)

# Print the shape of array
print(my2DArray.shape)

# Print out the data type of the array
print(my2DArray.dtype)

# Print the stride of the array.

# So what is stride of array?  It is the number of locations in memory between 
# beginnings of successive array elements, measured in bytes or in units of the
# size of the array's elements

print(my2DArray.strides)

# Array of ones
ones = np.ones((3,4))
print(ones)

# Array of zeros
zeros = np.zeros((2,3,4), dtype=np.int16)
print(zeros)

# Array of random numbers
np.random.random((2,3))

# Empty array
emptyArray = np.empty((2,3))
print(emptyArray)

# Full array
fullArray = np.full((2,2),7)
print(fullArray)

# Array of evely spaced values
evenSpacedArray = np.arange(10,25,5)
print(evenSpacedArray)

# Array of evely spaced values
evenSpacedArray2 = np.linspace(0,2,9)
print(evenSpacedArray2)

# Save a numpy array into a file
x = np.arange(0.0,50,1)
np.savetxt('./packt/data.out', x, delimiter=',')

# Loading numpy array from text
z = np.loadtxt('./packt/data.out', unpack=True)
print(z)

# Loading array using genfromtxt method
my_array2 = np.genfromtxt('./packt/data.out',
                          skip_header=1,
                          filling_values=-999)
print(my_array2)

# Print the number of dimensions
print(my2DArray.ndim)

# Print the number of elements
print(my2DArray.size)

# Print information about the memmory layout
print(my2DArray.flags)

# Print the length of an array in bytes
print(my1DArray.itemsize)

# Print total bytes consumed by the array
print(my2DArray.nbytes)


# Broadcasting in NumPy
# It permits NumPy to operate with arrays of different shapes

A = np.ones((6,8))
print(A.shape)
print(A)

B = np.random.random((6,8))
print(B.shape)
print(B)

# Sum of A and B, here the shape of both the matrix is same.
print(A + B)

# Two dimensions are also compatible when one of them is 1
x = np.ones((3,4))
print(x)
print(x.shape)

y = np.arange(4)
print(y)
print(y.shape)

x-y

# Arrays can be broadcasted if they are compatible in all dimensions
x = np.ones((6,8))
y = np.random.random((10,1,8))
print(x+y)
print(x)
print(x.shape)
print(y)
print(y.shape)

# Basic operations
x = np.array([[1,2,3],[2,3,4]])
y = np.array([[1,4,9],[2,3,-2]])
# Add two arrays
add = np.add(x,y)
print(add)
# Substract two arrays
sub = np.subtract(x,y)
print(sub)
# Multiply
mult = np.multiply(x,y)
print(mult)
# Divide
div = np.divide(x,y)
print(div)
# Calculated the remainder
rem = np.remainder(x, y)
print(rem)

# Slice, subset and index arrays
x = np.array([10, 20, 30, 40, 50])

# Select items at index 0 and 1
print(x[0:2])

# Select item at row 0 and 1 and column 1 from 2D array
y = np.array([[ 1,  2,  3,  4], [ 9, 10, 11 ,12]])
print(y[0:2, 1])

# Specifying conditions
biggerThan2 = (y >= 2)
print(y[biggerThan2])

# Import pandas
import pandas as pd

# Setting default parameters in pandas
print("Pandas Version:", pd.__version__)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Pandas works with series and dataframes
series = pd.Series([2, 3, 7, 11, 13, 17, 19, 23])
print(series)

# Creating dataframe from Dictionary
dict_df = [{'A': 'Apple', 'B': 'Ball'},{'A': 'Aeroplane', 'B': 'Bat', 'C': 'Cat'}]
dict_df = pd.DataFrame(dict_df)
print(dict_df)

# Creating dataframe from Series
series_df = pd.DataFrame({
    'A': range(1, 5),
    'B': pd.Timestamp('20190526'),
    'C': pd.Series(5, index=list(range(4)), dtype='float64'),
    'D': np.array([3] * 4, dtype='int64'),
    'E': pd.Categorical(["Depression", "Social Anxiety", "Bipolar Disorder", "Eating Disorder"]),
    'F': 'Mental health',
    'G': 'is challenging'
})
print(series_df)

# Creating a dataframe from ndarrays
sdf = {
    'County':['Østfold', 'Hordaland', 'Oslo', 'Hedmark', 'Oppland', 'Buskerud'],
    'ISO-Code':[1,2,3,4,5,6],
    'Area': [4180.69, 4917.94, 454.07, 27397.76, 25192.10, 14910.94],
    'Administrative centre': ["Sarpsborg", "Oslo", "City of Oslo", "Hamar", "Lillehammer", "Drammen"]
    }
sdf = pd.DataFrame(sdf)
print(sdf)

# Loading a dataset into a Pandas dataframe
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'ethnicity', 'gender','capital_gain','capital_loss','hours_per_week','country_of_origin','income']
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',names=columns)
df.head(10)
df.info()
df.shape
df.columns
df.describe()

# Selecting rows and columns
df.iloc[10]

# Select 10 rows
df.iloc[0:10]

# Select a range of rows
df.iloc[10:15]

# Select the last 2 rows
df.iloc[-2:]

# Select every other row in columns 3-5
df.iloc[::2,3:5].head()

np.random.seed(24)
df = pd.DataFrame({'F': np.linspace(1,10,10)})
df = pd.concat([df, pd.DataFrame(np.random.randn(10,5), columns=list('EDCBA'))],
               axis=1)
df.iloc[0,2] = np.nan
df

# Define a function that should color the values that are less than 0
def colorNegativeValueToRed(value):
  if value < 0:
      color = 'red'
  elif value > 0:
      color = 'black'
  else:
      color = 'green'
    
  return f'color: {color}'

styled = df.style.applymap(colorNegativeValueToRed, subset=['A','B','C','D','E'])
styled.to_html('./packt/styled_output.html')

# Let us hightlight max value in the column with green background and min value with orange background
def highlightMax(s):
    isMax = s == s.max()
    return ['background-color: orange' if v else '' for v in isMax]

def highlightMin(s):
    isMin = s == s.min()
    return ['background-color: green' if v else '' for v in isMin]

styled2 = df.style.apply(highlightMax).apply(highlightMin).highlight_null()
styled2.to_html('./packt/styled2_output.html')

# Import seaborn
import seaborn as sns


cm = sns.light_palette("pink", as_cmap=True)

styled3 = df.style.background_gradient(cmap=cm)
styled3.to_html('./packt/styled3_output.html')