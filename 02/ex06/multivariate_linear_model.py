import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

data = pd.read_csv("spacecraft_data.csv")
X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])
# print("================================================================")
# print("============================PART-ONE============================")
# print("================================================================\n\n")

# print("                              Age                              ")
# myLR_age = MyLR(theta = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
# print(f"initial thetas : {myLR_age.theta}")
# print(f"new thetas : {myLR_age.fit_(X[:,0].reshape(-1,1), Y)}")
# myLR_age.plot(X[:,0].reshape(-1,1),Y)
# y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
# print(f"MSE score:{myLR_age.mse_(y_pred,Y)}\n\n")

# X = np.array(data[['Thrust_power']])
# print("                          Thrust_power                          ")
# myLR_thrust = MyLR(theta = [[1000.0], [-1.0]], alpha = 0.00015, max_iter = 300000)
# print(f"initial thetas : {myLR_thrust.theta}")
# print(f"new thetas : {myLR_thrust.fit_(X[:,0].reshape(-1,1), Y)}")
# myLR_thrust.plot(X[:,0].reshape(-1,1),Y)
# y_pred = myLR_thrust.predict_(X[:,0].reshape(-1,1))
# print(f"MSE score:{myLR_thrust.mse_(y_pred,Y)}\n")

# X = np.array(data[['Terameters']])
# print("                           Terameters                           ")
# myLR_terameters = MyLR(theta = [[1000.0], [-1.0]], alpha = 0.00015, max_iter = 300000)
# print(f"initial thetas : {myLR_terameters.theta}")
# print(f"new thetas : {myLR_terameters.fit_(X[:,0].reshape(-1,1), Y)}")
# myLR_terameters.plot(X[:,0].reshape(-1,1),Y)
# y_pred = myLR_terameters.predict_(X[:,0].reshape(-1,1))
# print(f"MSE score:{myLR_terameters.mse_(y_pred,Y)}\n")

print("================================================================")
print("============================PART-TWO============================")
print("================================================================")
X = np.array(data[['Age','Thrust_power','Terameters']])
my_lreg = MyLR(theta = [[1.0], [1.0], [1.0], [1.0]], alpha = 1e-4, max_iter = 600000)

print(f"shape = {X.shape}")
y_pred = my_lreg.predict_(X)
print(f"MSE score:{my_lreg.mse_(y_pred,Y)}\n")
print(f"new thetas : {my_lreg.fit_(X, Y)}")
y_pred = my_lreg.predict_(X)
print(f"final MSE score:{my_lreg.mse_(y_pred,Y)}\n")
