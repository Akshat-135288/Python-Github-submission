import numpy as np

# Question 1: Array Creation

# 1.1 Create a 1D array of integers from 5 to 25
arr1D = np.arange(5, 26)   
print("1D Array (5 to 25):")
print(arr1D, "\n")


# 1.2 Create a 2D array with 3 rows and 4 columns filled with random integers between 10 and 50
arr2D = np.random.randint(10, 51, size=(3, 4))
print("2D Array (Random 3x4 between 10 and 50):")
print(arr2D, "\n")


# Question 2: Array Attributes

# 2.1 Attributes of 1D array
print("Attributes of 1D array:")
print("Shape:", arr1D.shape)
print("Size:", arr1D.size)
print("Data type:", arr1D.dtype, "\n")

# 2.2 Attributes of 2D array
print("Attributes of 2D array:")
print("Shape:", arr2D.shape)
print("Size:", arr2D.size)
print("Data type:", arr2D.dtype, "\n")



# Question 3: Array Operations

# 3.1 Create two 1D arrays
Array1 = np.array([2, 4, 6, 8, 10])
Array2 = np.array([1, 3, 5, 7, 9])

# 3.2 Perform operations
print("Array Operations:")
print("Addition:", Array1 + Array2)
print("Subtraction:", Array1 - Array2)
print("Element-wise Multiplication:", Array1 * Array2)
print("Element-wise Division:", Array1 / Array2, "\n")


# Question 4: Broadcasting

# 4.1 Create 2D array (3x3) with values 1 to 9
arr_broadcast = np.arange(1, 10).reshape(3, 3)
print("Original 3x3 Array:")
print(arr_broadcast, "\n")

# 4.2 Multiply with scalar 5 (broadcasting)
result = arr_broadcast * 5
print("After Broadcasting (multiplied by 5):")
print(result)
