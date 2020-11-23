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

# Importing the dataset 
dataset = pd.read_csv('Data.csv')

# Taking a quick look at the data structure
dataset.head()
dataset.info()
dataset["Date"].value_counts()

#Looking for standard correlations within Dataset
corr_matrix = dataset.corr()
corr_matrix["TargetPrice"].sort_values(ascending = False)

# Visualising scatter matrix (Focusing on the most promising attributes only)
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
groups = dataset["Date"] # will be used to split data into Training and Testing sets "Date"-wisely

# Function for Training and Validating a Regression Model
number_of_splits = 15 # the number of splits for model validation
def validate_regressor(regressor): 
    train_RMSE_scores = [] # array to store Train RMSE for all data splits
    test_RMSE_scores = [] # array to store Test RMSE for all data splits
    gss = GroupShuffleSplit(n_splits = number_of_splits, test_size = 0.2, random_state = 0)
    for train, test in gss.split(X, y, groups=groups): 
        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]
        # Fitting Regression Model to a specific Training Set
        regressor.fit(X_train, y_train)    
        # Calculating Regressor Training error
        regressor_train_mse = mean_squared_error(y_train, regressor.predict(X_train))
        train_RMSE_scores.append(np.sqrt(regressor_train_mse)) #storing Training RMSE in the array
        # Predicting results ona corresponding Test Set
        y_pred = regressor.predict(X_test)
        # Calculating Regressor Test error
        regressor_test_mse = mean_squared_error(y_test, y_pred)
        test_RMSE_scores.append(np.sqrt(regressor_test_mse)) #storing Test RMSE in the array
    # Printing out the results: 
    # average RMSE for all data splits and Standard Deviation of RMSE for all data splits
    print("Mean RMSE score for Training Sets: ", np.mean(train_RMSE_scores))
    print("Standard Deviation for Training Sets: ", np.std(train_RMSE_scores))
    print("Mean RMSE score for Test Sets: ", np.mean(test_RMSE_scores))
    print("Standard Deviation for Test Sets: ", np.std(test_RMSE_scores), "\n")
        
# Evaluating performance of Multiple Linear Regression Model on the Dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
validate_regressor(regressor)

# Evaluating performance of Decision Tree Regression Model on the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
validate_regressor(regressor)

# Evaluating performance of Random Forest Model on the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
validate_regressor(regressor)

# Evaluating performance of Linear Reression Model built on BidPrice and FeaturePrice4 only
X = dataset.iloc[:, [1,8]].values
regressor_2 = LinearRegression()
validate_regressor(regressor_2)

"""
Random Forest Model with 10 estimators demonstrated the best performance among the considered models: 
1) it produced the least RMSE on the training sets on average
2) it has rather close average Training RMSE and average Test RMSE
3) the deviations of Test RMSE values seem being acceptable
"""

# Training the chosen model (RandomForestRegressor) on the whole dataset
chosen_model = RandomForestRegressor(n_estimators = 10, random_state = 0)
X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:, 11].values
chosen_model.fit(X, y)

# Predicting the TargetPrice values for "Test.csv" data
control_dataset = pd.read_csv('Test.csv')
control_X = control_dataset.iloc[:, 1:11].values
y_predicted = chosen_model.predict(control_X)

# Saving Predicted TargetPrice values into a separate csv file
output = pd.DataFrame(y_predicted)
output.to_csv("Predicted_results.csv", header = ["Predicted TargetPrice"], index = False)


df = pd.read_csv('Predicted_results.csv')