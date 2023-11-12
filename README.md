## EXP NO. 02
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
### Date : 31.08.23
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.. 


## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by:D.Vinitha Naidu 
RegisterNumber: 212222230175 

```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

#Array value of X
X=df.iloc[:,:-1].values
X

#Array value of Y
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours Vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
### df.head()
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/94192dcb-de29-4fea-80bc-88e858c95ad8)

### df.tail()
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/f180113a-fc92-4790-b389-da0052ed8ba2)

### Array value of X
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/9da0c436-47c3-4f0a-9e39-3afa02325d75)

### Array value of Y
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/6bf0425f-3dc2-4cd1-acdb-0387199fcd6d)

### Values of Y prediction
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/feb32dbf-f127-4176-8485-bf0f8fbdb6a6)

### Array values of Y test
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/f0654024-6ccc-4c53-a6ae-17e896f1a007)

### Training Set Graph
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/6eb8a1a5-2684-4cd5-a2f4-ab091b8337e7)

### Test Set Graph
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/9d506a26-6479-4e33-9d10-19fd4268d616)

### Values of MSE, MAE and RMSE
![image](https://github.com/VinithaNaidu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166004/c115f593-01c2-480e-a34e-39b2d07250d9)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
