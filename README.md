# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import the dataset using pandas, clean column names if necessary, and separate input features (Size, Bedrooms) and target variables (Price, Occupants).
2. Apply feature scaling using StandardScaler to normalize the input features for better performance of the SGD regressor.
3. Initialize two SGDRegressor models and train them using the scaled features to predict house price and number of occupants separately.
4. Take user input (size and bedrooms), scale it using the same scaler, and use the trained models to predict the house price and number of occupants.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Ezhilan H
RegisterNumber:  212225240040

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("house.csv")
data.columns = data.columns.str.strip()

X = data[['Size', 'Bedrooms']]
y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))
new_data = scaler.transform([[size, bed]])
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
*/
```

## Output:
<img width="1224" height="798" alt="image" src="https://github.com/user-attachments/assets/4d12d042-a819-4f8b-8491-4eaf6898ce08" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
