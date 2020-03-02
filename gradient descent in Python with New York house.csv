import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('/Users/ruiwenzhang/Second Semester/Optimization/house.csv')
y = df['price']
X = df.drop(columns=['Unnamed: 0', 'price'])

#normalization for gradient descent to run faster
normalized_X=(X-X.mean())/X.std()
X = np.asmatrix(normalized_df.values)
X = np.hstack(((np.ones(len(X))).reshape(-1,1),X))
y = np.asmatrix(y.values)
y = y.reshape(-1,1)

#def costfunction(X,y,theta):
    m = np.size(y)

    #Cost function in vectorized form
    h = X @ theta
    J = float(sum((h - y).T @ (h - y))*(1./(2*m)));    
    return J;


def gradient_descent(X,y,theta,alpha = 0.01,num_iters=10, r = 0.2):
    #Initialisation of useful values 
    m = np.size(y)
    J_history = []
    theta_0_hist, theta_1_hist, theta_2_hist = [], [], [] #For plotting afterwards
    i = 0
    h = X @ theta
    #gradients= []
    while any((1/m)* (X.T @ (h-y)))>=r and i<num_iters:
        
        #Grad function in vectorized form
        gradient = (1/m)* (X.T @ (h-y))
        theta = theta - alpha *  gradient
        h = X @ theta

        #Cost and intermediate values for each iteration
        J_history.append(costfunction(X,y,theta))
        theta_0_hist.append(theta[0,0])
        theta_1_hist.append(theta[1,0])
        theta_2_hist.append(theta[2,0])
        #gradients.append(gradient)

        i = i+1
        
        #try with different learning rate
        #alpha = alpha/math.sqrt(i) 

    return theta,J_history,theta_0_hist, theta_1_hist, theta_2_hist
   
   
#plotting
    
#Setup of meshgrid of theta values
T0, T1 = np.meshgrid(np.linspace(0,111751*2,30),np.linspace(-20000,53302*3,30))#(np.linspace(20000,140000,30),np.linspace(20000,140000,30))
tt = [np.array([t0,t1]).reshape(-1,1) 
        for t0, t1 in zip(np.ravel(T0), np.ravel(T1))]
        
#Computing the gradient descent
theta_result,J_history, theta_0, theta_1, theta_2 = gradient_descent(X,y,np.array([659383,0,0]).reshape(-1,1),alpha = 0.01,num_iters=100,r = 0)

#plot cost history
%matplotlib inline
fig = plt.figure(figsize = (14,7))
ax = fig.add_subplot(1,1,1)
ax.plot(J_history,marker = '*')

#plot gradient and level set
#Angles needed for quiver plot
anglesx = np.array(theta_1)[1:] - np.array(theta_1)[:-1]
anglesy = np.array(theta_2)[1:] - np.array(theta_2)[:-1]

%matplotlib inline
fig = plt.figure(figsize = (16,8))
#Surface plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(T0, T1, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
ax.plot(theta_1,theta_2,J_history, marker = '*', color = 'r', alpha = .8, label = 'Gradient descent')

ax.set_xlabel('theta 1')
ax.set_ylabel('theta 2')
ax.set_zlabel('Cost function')
ax.set_title('Gradient descent: Root at {}'.format(theta_result.ravel()))
ax.view_init(45, 45)

#Contour plot
ax = fig.add_subplot(1, 2, 2)
ax.contour(T0, T1, Z, 70, cmap = 'jet')
ax.quiver(theta_1[:-1], theta_2[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .9)

plt.show()
