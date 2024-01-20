import numpy as np
from tools import add_intercept

x = np.arange(1,7).reshape(-1, 2)
newone = add_intercept(x)
print(newone)

y = np.array([[]])
newone = add_intercept(y)
print(newone)

y = np.array([[1, 3, 4 ,5 ,6]])
newone = add_intercept(y)
print(newone)


y = np.arange(1,10).reshape((3,3))
newone = add_intercept(y)
print(newone)