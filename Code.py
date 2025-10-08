#Import Importent Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Read Train and Test Dataset
train_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

#Describe Train Dataset
print("Train Data:")
train_data.describe()

#Describe Test Dataset
print("Test Data:")
test_data.describe()

#Taking the Main Variables
price = train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
factors = train_data[features]
factors_test = test_data[features]

#Dividing Train Data
train_X, val_X, train_y, val_y = train_test_split(factors, price, random_state=1)

#Get Mean Absolute Error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

#Finding Minimum MAE to take the Best Leaf Size
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

#Final Model Training
final_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(factors,price)

#Predictions for Test Dataset
test_preds = final_model.predict(factors_test)

#Creating Submission file in .csv format
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
