from z_score import zscore
import numpy

# Example 1:
X = numpy.array([0, 15, -9, 7, 12, 3, -21])
print("test 1")
print(zscore(X))

# Example 2:
Y = numpy.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print("test 2")
print(zscore(Y))
