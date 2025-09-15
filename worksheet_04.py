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



# Question 5: Slicing and Indexing
arr4 = np.arange(10, 26).reshape(4, 4)
print("\n5.1 4x4 Array:\n", arr4)
print("5.2 Second Row:", arr4[1])
print("5.2 Last Column:", arr4[:, -1])
arr4[0] = 0
print("5.3 First row replaced with zeros:\n", arr4)



# Question 6: Boolean Indexing
arr5 = np.random.randint(20, 41, size=10)
print("\n6.1 Random Array:", arr5)
print("6.2 Elements > 30:", arr5[arr5 > 30])



# Question 7: Reshaping
arr6 = np.arange(11, 23)
reshaped_arr6 = arr6.reshape(3, 4)
print("\n7.1 1D Array:", arr6)
print("7.2 Reshaped to 3x4:\n", reshaped_arr6)



# Question 8: Matrix Operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("\n8.1 Matrices")
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("8.2 Matrix Multiplication:\n", A @ B)
print("Transpose of A:\n", A.T)




# Question 9: Statistical Functions
arr7 = np.random.randint(10, 61, size=15)
print("\n9.1 Random Array:", arr7)
print("Mean:", np.mean(arr7))
print("Median:", np.median(arr7))
print("Standard Deviation:", np.std(arr7))




# Question 10: Linear Algebra
A = np.array([[2, 1, 3],
              [0, 5, 6],
              [7, 8, 9]])
print("\n10.1 Matrix A:\n", A)
print("Determinant:", np.linalg.det(A))
print("Inverse:\n", np.linalg.inv(A))
eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)




# Question 11: Robot Path
positions = np.array([[0, 0], [2, 3], [4, 7], [7, 10], [10, 15]])


# 11.1 Euclidean distance between first and last
dist_first_last = np.linalg.norm(positions[-1] - positions[0])
print("\n11.1 Euclidean Distance:", dist_first_last)

# 11.2 Total distance step by step
step_dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
total_dist = np.sum(step_dists)
print("11.2 Total Distance:", total_dist)



