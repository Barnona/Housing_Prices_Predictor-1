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
