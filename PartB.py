import csv 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Input data
data = list(csv.reader(open("data.csv")))
data = pd.read_csv('data.csv')
A = data.iloc[:, :2]
a = data.iloc[:,1:2]
print(type(a))
b = data.iloc[:,-1:]

#initialize guess a1 and a0 x_ = [a0 a1]

x_ = np.array([[0],[0]])
a0 = x_[0]
a1 = x_[1]

#calculate d
Ax = np.dot(A,x_) 
r = Ax - b #r_ = Ax_ - b_ 
At = np.transpose(A) #AˆT
gradient = 2 * np.dot(At,r) #2.AˆT.r_
d = -gradient #d=gradient at x_ = [0 0] 

# The fixed step size alpha
alpha = 0.00000001 
max_iteration = 1000  # The max number of iteration

#analytic alpha
Art = np.dot(np.transpose(r),A) #A.rˆT
alpha = np.dot(Art,d) / (np.dot(A,d))**2

for i in range(max_iteration): 
    Y_pred = -a0 + a1*a #keep track with the regression line  
    d = -2 * np.dot(At,np.dot(A,x_)-b)             # The predicted value of Y
    x_ = x_ + alpha*d #update x_ aka [a0 a1]
    plt.plot([min(a), max(a)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
    plt.plot(x_[0],x_[1])
    
plt.scatter(a, b) 
print(-x_[0],x_[1])


plt.show()