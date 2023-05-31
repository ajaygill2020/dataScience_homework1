# Ajay Gill
# MIS 6900 - Summer 2023
# 25 May 2023

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

housing_df = pd.read_csv("./data/kc_house_data_original.csv")

housing_df.info()

housing_df.head()


# Data prep steps
print(int(True), int(False))

housing_df['price_gt_1M'] = housing_df['price'].map(lambda x: int(x >= 1000000)) 

housing_df = housing_df.iloc[:, 2:]
housing_df.info()

newcols_class = [_ for _ in range(1, 20)]
newcols_class

newcols_regression = [_ for _ in range(1, 18)]
newcols_regression.extend([0])
newcols_regression

housing_class_df = housing_df.iloc[:, newcols_class]
housing_class_df.info()

housing_regression_df = housing_df.iloc[:, newcols_regression]
housing_regression_df.info()

housing_class_df.to_csv("./data/kc_house_data_classification.csv", index=False)
housing_regression_df.to_csv("./data/kc_house_data_regression.csv", index=False)


# EDA using pandas
from pandas_profiling import ProfileReport
profile = ProfileReport(train_df, title="Pandas Profiling Report")
profile.to_file("output/pandas_profiling_report.html")