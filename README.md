
# Real Estate Price Prediction

## Project Description
This project aims to estimate or forecast future real estate property prices using a dataset of various property features. The goal is to provide accurate property valuations to help buyers, sellers, investors, and real estate professionals make informed decisions.

## Table of Contents
1.  [Installation](#installation)
2.  [Dataset](#dataset)
3.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4.  [Model Building](#model-building)
5.  [Model Evaluation](#model-evaluation)

## Installation
To run this notebook, you'll need the following libraries. You can install them using pip:

```python
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset
The dataset `Real_Estate.csv` contains information about real estate properties, including transaction date, house age, distance to the nearest MRT station, number of convenience stores, latitude, longitude, and the house price of the unit area.

### Load Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("/content/Real_Estate.csv")

# Display the first few rows of the dataset
display(data.head())
```

### Dataset Information
```python
data.info()
```

### Dataset Shape
```python
data.shape
```

### Basic Statistical Info
```python
data.describe()
```

### Missing Values
```python
data.isnull().sum()
```
*Observation: There are no missing values in the dataset.*

## Exploratory Data Analysis (EDA)

### Histograms of Numerical Features
Histograms were generated to understand the distribution of each numerical variable.

```python
sns.set_style("whitegrid")

cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores',
        'Latitude', 'Longitude', 'House price of unit area']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig.suptitle('Histograms of Real Estate Data', fontsize=16)

for i, col in enumerate(cols):
    sns.histplot(data[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(col)
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

### Scatter Plots with House Price
Scatter plots were used to visualize the relationships between features and the target variable (`House price of unit area`).

```python
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)

sns.scatterplot(data=data, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(data=data, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(data=data, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(data=data, x='Latitude', y='House price of unit area', ax=axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

### Correlation Matrix
A correlation matrix was used to quantify the relationships between variables.

```python
correlation_matrix = data.drop('Transaction date', axis=1).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

print(correlation_matrix)
```
*Observation: 'Distance to the nearest MRT station' shows a strong negative correlation (-0.637) with house price, and 'Number of convenience stores' shows a moderate positive correlation (0.281).*

## Model Building

A Linear Regression model was built to predict real estate prices.

```python
# Selecting features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

X = data[features]
y = data[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)
```

## Model Evaluation

### Actual vs. Predicted Values Plot
This plot visualizes how well the model's predictions align with the actual house prices.

```python
y_pred_lr = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
plt.show()
```

### Mean Squared Error (MSE)
```python
y_predict = model.predict(X_test)
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_predict)}")
```

### Root Mean Squared Error (RMSE)
```python
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print(f"Root Mean Squared Error (RMSE): {rmse}")
```

