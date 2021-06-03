import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df_wine = pd.read_csv("E:\\DataMining\\PredictableWinePrice\\WineData.csv")

points = df_wine["points"]
# points = df_wine["points"]
# points = points.head(200)

price = df_wine["price"]
# price = df_wine["price"]
# price = price.head(200)

points_train, points_test, price_train, price_test = train_test_split(points.values.reshape(-1,1), price.values.reshape(-1,1), test_size=0.3, random_state=35)
regr_train = linear_model.LinearRegression().fit(points_train, price_train)
price_pred = regr_train.fit(points_train, price_train).predict(points_test)
price_train_pred = regr_train.predict(points_train)

plt.scatter(points_train, price_train, color='green')
plt.scatter(points_train, price_train_pred, color='red')
plt.scatter(points_test[:10,:], price_test[:10], color='black')
plt.title('Data Classification')
plt.xlabel('Points')
plt.ylabel('Price')
plt.show()

regr = linear_model.LinearRegression()
regr.fit(points.values.reshape(-1,1),price)
plt.plot(points, regr.predict(points.values.reshape(-1,1)), color = "blue")
plt.xlabel("Points")
plt.ylabel("Price")
plt.scatter(points, price, color = "red")
plt.show()

plt.plot([min(price_pred), max(price_pred)],[min(price_test),max(price_test)])
plt.scatter(price_pred, price_test, color='red')
plt.title('Compare')
plt.xlabel('Price_pred')
plt.ylabel('Price_test')
plt.show()

#Tính sai số bình phương trung bình
print('Coefficients: \n', regr.coef_)
print('Bias: \n', regr.intercept_)
print('Mean squared error: %.2f'
% mean_squared_error(price_test, price_pred))
#Hệ số xác định
#Biểu diễn sự phù hợp của mô hình hồi quy tuyến tính khi áp dụng trong bộ dữ liệu này
print('Coefficient of determination: %.2f'
% r2_score(price_test, price_pred))

#Dự đoán giá rượu
print('Point of wine: ')
need_prediction = input()
need_prediction = float(need_prediction)
pricePredict = regr.predict([[need_prediction]])
print("Price of wine predict: ", pricePredict)
 

