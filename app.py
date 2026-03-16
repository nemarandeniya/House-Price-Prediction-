import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

houseprice_dataframe = pd.read_csv("housing.csv")
# print(houseprice_dataframe)

houseprice_dataframe['price'] = houseprice_dataframe['median_house_value']
#print(houseprice_dataframe.head())# first 5 row
# print(houseprice_dataframe.shape)#check rows and columns


# check for missing value---------------------------------
# print(houseprice_dataframe.isnull().sum())

houseprice_dataframe['total_bedrooms'] = houseprice_dataframe['total_bedrooms'].fillna(houseprice_dataframe['total_bedrooms'].median())
# print(houseprice_dataframe.isnull().sum())

# print(houseprice_dataframe.describe())


# understanding the correlation---------------------------
numeric_df = houseprice_dataframe.select_dtypes(include=np.number)
correlation = numeric_df.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.title("Housing Dataset Feature Correlation",fontsize=13)
#plt.show()


# sns.heatmap() → draws the heatmap from the correlation matrix.
# cbar=True → shows the color bar on the side to indicate correlation values.
# square=True → makes each cell a square, so the grid looks neat.
# fmt='.1f' → formats numbers to one decimal place.
# annot=True → prints the correlation values inside each square.
# annot_kws={'size':8} → sets annotation font size to 8.
# cmap='Blues' → sets a blue color scheme: darker blue → higher correlation, lighter → lower.


X = houseprice_dataframe.drop(['price'],axis=1)
Y = houseprice_dataframe['price']

# print(X)
# print(Y)

# Spliiting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape,X_train.shape, X_test.shape)

# Model training
model = XGBRegressor()
model.fit(X_train,Y_train)

# prediction on train data-----------------

training_data_prediction = model.predict(X_train)

# R squared error
score_1 = metrics.r2_score(Y_train,training_data_prediction)

# Mean Absolute error
score_2 = metrics.mean_absolute_error(Y_train,training_data_prediction)# find the difference between original data & predict data

print("R squared error :" ,score_1)
print("Mean absolute error :" ,score_2)

#prediction on test data----------------
test_data_prediction = model.predict(X_test)

# R squared error
score_1 = metrics.r2_score(Y_test,test_data_prediction)

# Mean Absolute error
score_2 = metrics.mean_absolute_error(Y_test,test_data_prediction)# find the difference between original data & predict data

print("R squared error :" ,score_1)
print("Mean absolute error :" ,score_2)

# Visualize the actual prices and predicted prices
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual price vs Predicted price")
plt.show()