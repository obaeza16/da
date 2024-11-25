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

