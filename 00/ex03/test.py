import numpy as np
from tools import add_intercept

x = np.arange(1,6)
newone = add_intercept(x)
print(newone)

y = np.arange(1,10).reshape((3,3))
newone = add_intercept(y)
print(newone)