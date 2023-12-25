# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:54:11 2023

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('‏‏Salary_Data.csv')
print(data)

print(data.describe())

print(data.head())
plt.scatter(data['YearsExperience'],data['Salary'])
plt.show()

#y=mx+b

print(data.head())

x=data.iloc[:,:1]  
y=data.iloc[:,1]

print(x)
print(y)

from sklearn.linear_model import  LinearRegression
model =LinearRegression()
model.fit(x,y)
#m
print(model.coef_)
#b
print(model.intercept_)

#salary(y)=model.coef_ * YearsExperience(x) + model.intercept_



plt.scatter(x,y)
plt.plot(x,model.predict(x),'r')

model.predict([[2]])

#pridict
model.predict([[24]])
model.predict([[12]])
model.predict([[100]])







#valdation
model.score(x,y)
