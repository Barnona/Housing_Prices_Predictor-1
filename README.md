# Introduction
After completing the course [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning), I'm submitting this notebook to this competition. I know it's just a basic of ML, but hey, why not test my knowledge!!

# Import Importent Libraries
* **Pandas**: Pandas as pd. This library is used in r/w on I/O files
* **RandomForestRegressor**: from *sklearn.ensemble* we import RandomForestRegressor. An estimator for modal training (here, for main use)
* **mean_absolute_error**: from *sklearn.metrics* import mean_absolute_error
* **train_test_split**: from *sklearn.model_selection* import train_test_split
* **DecisionTreeRegressor**: from *sklearn.tree* import DecisionTreeRegressor. An estimator for modal training (here, for initial use)

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
```

# Read the .csv files and Describe

**Read Train and Test Dataset**

```python
train_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
```

**Describe Train Dataset**

```python
print("Train Data:")
train_data.describe()
```

**Describe Test Dataset**

```python
print("Test Data:")
test_data.describe()
```

# Taking the Main Variables
"price" for Sale Price in Train Dataset and "factors" for some features in Train Dataset. We are also taking another variable, called "factors_test" for same features in Test Dataset

```python
price = train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
factors = train_data[features]
factors_test = test_data[features]
```

# To Find Best Leaf Size
Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. Here we are initiating a model using **DecisionTreeRegressor**

1. **Dividing Train Data**

```python
train_X, val_X, train_y, val_y = train_test_split(factors, price, random_state=1)
```

2. **Get Mean Absolute Error**

```python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

3. **Finding Minimum MAE to take the Best Leaf Size**

```python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
```

# Final Model Training
After finding the best tree size, we are gonna train our final model using **RandomForestRegressor**

```python
final_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(factors,price)
```

# Model Predictions and Submission

```python
#Predictions for Test Dataset
test_preds = final_model.predict(factors_test)

#Creating Submission file in .csv format
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```
