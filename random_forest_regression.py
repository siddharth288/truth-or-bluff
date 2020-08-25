# Random Forest Regression

# importing libraries
import numpy as np #for mathematical functions
import matplotlib.pyplot as plt # for plotting graphs
import pandas as pd # pd is the short form, importing datasets

# importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# splitting the dataset into training set and test data
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0 )"""

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# fitting the regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, criterion="mse", random_state = 0)
regressor.fit(x, y)

# predicting a new result
y_pred = regressor.predict( [[6.5]] )

# visualizing the regression results( for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid) , 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("truth or bluff (Random forest regression model)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()