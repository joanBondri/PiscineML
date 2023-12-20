from TinyStatistician import TinyStatistician
import numpy as np

yop = TinyStatistician()

list = [1, 2, 4, 8, 16, 34]

array = np.array(list)

print(yop.mean(list))
print(yop.mean(array))
print(yop.mean("ca va pas"))