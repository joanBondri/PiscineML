from TinyStatistician import TinyStatistician
import numpy as np

yop = TinyStatistician()

list = [1, 2, 4, 8, 16, 34]

array = np.array(list)

emptyArray = np.array([])
uniqueArray = np.array([4])

print(yop.mean(list))
print(yop.mean(array))
# print(yop.mean("ca va pas"))

print(yop.median(array))
print(yop.median(emptyArray))
print(yop.median(uniqueArray))
print(yop.var(array))
print(yop.std(array))
print(yop.quartile(array))

print(yop.percentile([1, 3, 23, 45, 67, 78, 89], 51))

a = [1, 42, 300, 10, 59]
print("mmh1")
print(TinyStatistician().mean(a))
print(TinyStatistician().median(a))
print(TinyStatistician().quartile(a))
print(f"10th : {TinyStatistician().percentile(a, 10)}")
print(TinyStatistician().percentile(a, 15))
print(TinyStatistician().percentile(a, 20))
print(TinyStatistician().var(a))
print(TinyStatistician().std(a))
print("mmh2")
