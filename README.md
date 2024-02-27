# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn. 
4. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divyadharshini A
RegisterNumber:  212222240027
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![Screenshot 2024-02-27 092230](https://github.com/Senthamil1412/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119120228/6bcaca0b-9cee-4dbf-9c78-674731a01645)
![Screenshot 2024-02-27 093132](https://github.com/Senthamil1412/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119120228/bb792d3b-9636-47d7-9b7f-4432074573e4)
![Screenshot 2024-02-27 092636](https://github.com/Senthamil1412/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119120228/514c4be3-3c77-4d0d-9d3f-7193b9257250)
![Screenshot 2024-02-27 092650](https://github.com/Senthamil1412/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119120228/bf71a9ea-c5ba-4458-971b-72678dad9120)
![Screenshot 2024-02-27 092720](https://github.com/Senthamil1412/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119120228/8803343f-51fa-4284-aace-cd8316b56d01)
![Screenshot 2024-02-27 092727](https://github.com/Senthamil1412/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119120228/7136c9ea-516d-48f6-a118-4365191298e0)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
