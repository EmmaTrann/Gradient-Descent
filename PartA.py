import pandas as pd
import matplotlib.pyplot as plt

#Input data
data = pd.read_csv('data.csv')
X = data.iloc[:,1]
Y = data.iloc[:,2]


#initialize a1 and a0
a0 = 0
a1 = 0

alpha = 0.0001  # The fixed step size alpha
max_iteration = 6  # The max number of iteration

n = float(len(X)) # length of X
al1=[]
al0=[]
for i in range(max_iteration): 
    Y_pred = a0 + a1*X  # The predicted value of Y
    
    #sum (yi - (a0+a1x)**2) derivative wrt to a1
    D_a1 = (2/n) * sum(X * (Y - Y_pred))  # Gradient - Derivative wrt a0
    #sum (yi - (a0+a1x)**2) derivative wrt to a0
    D_a0 = -(2/n) * sum(Y - Y_pred)  # Gradient - Derivative wrt a1
    a0 = a0 + alpha * D_a0  # Update a0
    al0.append(a0)
    a1 = a1 + alpha * D_a1  # Update a1
    al1.append(a1)
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
    

print(a0)
print(a1)
#plt.plot(al0,al1)
Y_pred = a0 + a1*X
plt.scatter(X, Y) 
#plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()