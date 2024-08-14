
# Solubility Prediction Using Linear Regression

## Overview
This project aims to predict the solubility (`logS`) of chemical compounds using a linear regression model. The dataset used contains features such as **MolLogP**, **MolWt**, **NumRotatableBonds**, and **AromaticProportion** to predict the `logS` (solubility) value.

## Dataset
The dataset, `delaney.csv`, consists of 1144 rows and 5 columns:

- **MolLogP**: Logarithm of the partition coefficient between octanol and water.
- **MolWt**: Molecular weight of the compound.
- **NumRotatableBonds**: Number of rotatable bonds in the molecule.
- **AromaticProportion**: Proportion of aromatic bonds in the molecule.
- **logS**: Logarithm of the solubility in mols per liter (target variable).

### Sample of the Dataset:
| MolLogP | MolWt  | NumRotatableBonds | AromaticProportion | logS  |
|---------|--------|-------------------|--------------------|-------|
| 2.5954  | 167.85 | 0.0               | 0.0                | -2.18 |
| 2.3765  | 133.405| 0.0               | 0.0                | -2.00 |
| 2.5938  | 167.85 | 1.0               | 0.0                | -1.74 |
| 2.0289  | 133.405| 1.0               | 0.0                | -1.48 |
| 2.9189  | 187.375| 1.0               | 0.0                | -3.04 |

## Data Preprocessing
The data preprocessing involved separating the features from the target variable and splitting the dataset into training and testing sets:

```python
y = df['logS']
x = df.drop('logS', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
```

## Model Building
A linear regression model was built using the training data:

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
```

## Predictions
Predictions were made on both the training and testing sets:

```python
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
```

## Data Normalization
To better compare the actual and predicted values, the data was normalized:

```python
normalized_y_train = (y_train - min_y_train) / (max_y_train - min_y_train)
normalized_y_test = (y_test - min_y_test) / (max_y_test - min_y_test)

normalized_y_train_pred = (y_lr_train_pred - min_y_lr_train_pred) / (max_y_lr_train_pred - min_y_lr_train_pred)
normalized_y_test_pred = (y_lr_test_pred - min_y_lr_test_pred) / (max_y_lr_test_pred - min_y_lr_test_pred)
```

## Model Evaluation
The model's performance was evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics:

```python
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(normalized_y_train, normalized_y_train_pred)
lr_test_mse = mean_squared_error(normalized_y_test, normalized_y_test_pred)

lr_train_r2 = r2_score(normalized_y_train, normalized_y_train_pred)
lr_test_r2 = r2_score(normalized_y_test, normalized_y_test_pred)

print('lr_train_mse:', lr_train_mse)
print('lr_test_mse:', lr_test_mse)
print('lr_train_r2:', lr_train_r2)
print('lr_test_r2:', lr_test_r2)
```

### Results:
- **Train MSE**: 0.0095
- **Test MSE**: 0.0128
- **Train R²**: 0.735
- **Test R²**: 0.572

## Conclusion
The linear regression model performed well on the training data with an R² of 0.735, explaining 73.5% of the variance in solubility. However, the model's performance on the test data was slightly lower with an R² of 0.572. While the model generalizes reasonably well, there may be room for improvement in future iterations.
