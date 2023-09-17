import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import numpy
from numpy import array 
from sklearn import preprocessing
dataset = pd.read_csv("Simulation_Results_Efficiency.csv")
dataset.head()

#Setting the value for X and Y
x = dataset[['Arc Center Distance', 'Diameter', 'Width', 'Pitch', 'Blade Count', 'Rotational Speed']]
y = dataset['Efficiency']

#normalizing data
x = 10*preprocessing.normalize(x)

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

#Fitting the Multiple Linear Regression model
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train)

#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:", mlr.coef_)
list(zip(x, mlr.coef_))

#Prediction of test set
y_pred_mlr= mlr.predict(x_test)

#Actual value and the predicted value
slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})

#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

import statsmodels.api as sm

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

p_values = model.summary2().tables[1]['P>|t|']
print('p values', p_values)

plt.scatter(y_pred_mlr, y_test)
