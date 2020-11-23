"""
TEST ASSIGNMENT for Atto Trading Technologies LLC
Author: Iryna Savchuk
Usage: Internal 
Version: 1.0
Date: 1 August 2018 
"""

# Importing the packages and libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

# Importing the datasets
dataset = pd.read_csv('Data.csv')
control_dataset = pd.read_csv('Test.csv')

# Taking a quick look at the data structure
dataset.head()
dataset.info()
dataset["Date"].value_counts()

#Looking for standard correlations within Dataset
corr_matrix = dataset.corr()
corr_matrix["TargetPrice"].sort_values(ascending = False)

# Visualising scatter matris (Focusing on the most promising attributes)
from pandas.tools.plotting import scatter_matrix
attributes = ["TargetPrice","Bid_Price","FeaturePrice4","FeaturePrice1","FeaturePrice2","Ask_Shares"]
scatter_matrix(dataset[attributes])

# Zooming the most promising attributes to predict the TargetPrice
dataset.plot(kind = "scatter", x = "Bid_Price", y = "TargetPrice", alpha = 0.1)
dataset.plot(kind = "scatter", x = "FeaturePrice4", y = "TargetPrice", alpha = 0.1)
dataset.plot(kind = "scatter", x = "Ask_Shares", y = "TargetPrice", alpha = 0.1)

# Preparing Data
X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:, 11].values
groups = dataset["Date"]

"""
# Training the Model on Group Shuffle Split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
number_of_splits = 10
train_MSE_scores = []
test_MSE_scores = []
gss = GroupShuffleSplit(n_splits = number_of_splits, test_size = 0.2, random_state = 0)
for train, test in gss.split(X, y, groups=groups): 
   # print("%s %s" % (train, test))
    X_train = X[train, :]
    y_train = y[train]
    X_test = X[test, :]
    y_test = y[test]
    # Fitting Regressor to the Training set
    regressor.fit(X_train, y_train)    
    # Evaluating model's training MSE
    regressor_train_mse = mean_squared_error(y_train, regressor.predict(X_train))
    train_MSE_scores.append(regressor_train_mse)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    #y_pred = sc_y.inverse_transform(regressor.predict(X_test))
    regressor_test_mse = mean_squared_error(y_test, y_pred)
    test_MSE_scores.append(regressor_test_mse)
    
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", np.mean(scores))
    print("Standard Devioation: ", np.std(scores))
    
display_scores(train_MSE_scores)
display_scores(test_MSE_scores)
"""

"""def display_scores(scores):
    print("Mean RMSE score: ", np.mean(scores))
    print("Standard Deviation: ", np.std(scores))
    
display_scores(train_MSE_scores)
display_scores(test_MSE_scores)"""

# Training the Model on Group Shuffle Split

def validate_regressor(regressor): 
    number_of_splits = 10
    train_RMSE_scores = []
    test_RMSE_scores = []
    gss = GroupShuffleSplit(n_splits = number_of_splits, test_size = 0.2, random_state = 0)
    for train, test in gss.split(X, y, groups=groups): 
        # print("%s %s" % (train, test))
        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]
        # Fitting Regressor to the Training set
        regressor.fit(X_train, y_train)    
        # Evaluating model's training RMSE
        regressor_train_mse = mean_squared_error(y_train, regressor.predict(X_train))
        train_RMSE_scores.append(np.sqrt(regressor_train_mse))
        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        regressor_test_mse = mean_squared_error(y_test, y_pred)
        test_RMSE_scores.append(np.sqrt(regressor_test_mse))
    print("Mean RMSE score for Training Sets: ", np.mean(train_RMSE_scores))
    print("Standard Deviation: ", np.std(train_RMSE_scores))
    print("Mean RMSE score for Test Sets: ", np.mean(test_RMSE_scores))
    print("Standard Deviation: ", np.std(test_RMSE_scores))
    #return([train_RMSE_scores,test_RMSE_scores])
        
# Evaluating Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
validate_regressor(regressor)

# Evaluating Decision Tree Regression Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
validate_regressor(regressor)

# Evaluating Random Forest Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
validate_regressor(regressor)

"""
# Evaluating SVR Model (with prior Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)
from sklearn.svm import SVR
regressor = SVR(kernel ='rbf')
train_RMSE_scores = []
test_RMSE_scores = []
gss = GroupShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
for train, test in gss.split(X, y, groups=groups): 
        X_train = X_scaled[train, :]
        y_train = y_scaled[train]
        X_test = X_scaled[test, :]
        y_test = y[test]
        # Fitting Regressor to the Training set
        regressor.fit(X_train, y_train)    
        # Evaluating model's training RMSE
        regressor_train_mse = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(regressor.predict(X_train)))
        train_RMSE_scores.append(np.sqrt(regressor_train_mse))
        # Predicting the Test set results
        y_pred = sc_y.inverse_transform(regressor.predict(X_test))
        regressor_test_mse = mean_squared_error(y_test, y_pred)
        test_RMSE_scores.append(np.sqrt(regressor_test_mse))
print("Mean RMSE score for Training Sets: ", np.mean(train_RMSE_scores))
print("Standard Deviation: ", np.std(train_RMSE_scores))
print("Mean RMSE score for Test Sets: ", np.mean(test_RMSE_scores))
print("Standard Deviation: ", np.std(test_RMSE_scores))
"""
"""

import statsmodels.formula.api as sm

gss = GroupShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
for train, test in gss.split(X, y, groups=groups): 
        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]
model = sm.OLS(y_train,X_train)
results = model.fit()
results.params"""

model = LinearRegression()
model.fit(X, y)
TargetPrice_predicted = model.predict(test_dataset.iloc[:, 1:11].values)
