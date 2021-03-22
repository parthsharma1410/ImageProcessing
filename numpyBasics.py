#19BCE0761 Parth Sharma
#Activity2 Numpy and Pandas

#importing numpy
import numpy as np
#importing linear algebra module of numpy
import numpy.linalg as lin

#Q1 - Reshaping matrices
#initializing x with array elements
x = np.array([4, 6, 2, 0, 1, 5, 0, 3, 2])
#creating a 3x3 matrix A (given in the question) by reshaping x
A = np.reshape(x,(3,3))
print("Matrix A: (3x3)")
print(A)
#reshaping it into a 1x9 matrix
y = np.reshape(x,(1,9))
print("Reshaped matrix A: (1x9)")
print(y)
#initializing z with array elements
z = np.array([0, 1, -1, 3, -1, 4, -1, 2, 1])
#creating a 3x3 matrix B (given in the question) by reshaping z
B = np.reshape(z,(3,3))
print("Matrix B: (3x3)")
print(B)
#reshaping it into a 1x9 matrix
w = np.reshape(z,(1,9))
print("Reshaped matrix B: (1x9)")
print(w)

#Q2 - Find maximum minimum values in matrix A
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
#Finding max element in Matrix A
max = np.max(A)
#Finding min element in Matrix A
min = np.min(A)
print("Max element in A:")
print(max)
print("Min element in A:")
print(min)

#Q3 - Find transpose matrix of matrix A and matrix B
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
#Finding transpose of A
atrans = A.T
print("Transpose of Matrix A:")
print(atrans)
#creating a 3x3 matrix B (given in the question)
B = np.array([[0, 1, -1], [3, -1, 4], [-1, 2, 1]])
print("Matrix B: (3x3)")
print(B)
#Finding transpose of B
btrans = B.T
print("Transpose of Matrix B:")
print(btrans)

#Q4 - Transform a matrix A into a one-dimensional array
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
#making a 1D array out of A
a1D = A.flatten()
print("1D array made from Matrix A:")
print(a1D)

#Q5 - Finding determinant of Matrix A
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
#Finding determinant of Matrix A
detA = lin.det(A)
print("Determinant of Matrix A:", detA)

#Q6 - Finding diagonal elements of Matrix A and Matrix B
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
diagA = A.diagonal()
print("Diagonal elements of Matrix A:", diagA)
#creating a 3x3 matrix B (given in the question)
B = np.array([[0, 1, -1], [3, -1, 4], [-1, 2, 1]])
print("Matrix B: (3x3)")
print(B)
diagB = B.diagonal()
print("Diagonal elements of Matrix B:", diagB)

#Q7 - Addition and subtraction of Matrix A and Matrix B
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
#creating a 3x3 matrix B (given in the question)
B = np.array([[0, 1, -1], [3, -1, 4], [-1, 2, 1]])
print("Matrix B: (3x3)")
print(B)
#Sum of Matrices A and B
sum = np.add(A,B)
print("Sum of Matrices A and B:")
print(sum)
#Difference of Matrices A and B
diff = np.subtract(A,B)
print("Difference of Matrices A and B:")
print(diff)

#Q8 - Multiplication of Matrices A and B
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
#creating a 3x3 matrix B (given in the question)
B = np.array([[0, 1, -1], [3, -1, 4], [-1, 2, 1]])
print("Matrix B: (3x3)")
print(B)
#Product of Matrices A and B
prod = np.matmul(A,B)
print("Product of Matrices A and B:")
print(prod)

#Q9 - Calculating inverse of Matrix A
#creating a 3x3 matrix A (given in the question)
A = np.array([[4, 6, 2], [0, 1, 5], [0, 3, 2]])
print("Matrix A: (3x3)")
print(A)
#Finding inverse of Matrix A
invA = lin.inv(A)
print("Inverse of Matrix A:")
print(invA)