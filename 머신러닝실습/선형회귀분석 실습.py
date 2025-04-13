import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

data = fetch_california_housing(as_frame= True)
df = data.frame
df.head()


X = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = \
train_test_split(X,y,test_size=0.2, random_state=24)

model = LinearRegression()
model.fit(X_train, y_train) #모수추정, 적합
coefficients = pd.DataFrame({
    "Feature" : X.columns,
    "Coefficient": model.coef_})
print(coefficients)
print(model.intercept_)

#추정된 회귀계수 수식으로 직접 구해보기
from sklearn.preprocessing import add_dummy_feature
X_train1 = add_dummy_feature(X_train)
coef = np.linalg.inv(X_train1.T @ X_train1) @ X_train1.T @ y_train
coefficients1 = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coef[1:] #0번째가 빠져있음. 앞과 똑같은 형식으로 출력하기 위해서
    })
print(coefficients1)
print(coef[0])#베타0

#예측성능(MSE) 확인
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error (MSE): {mse:2f}")
