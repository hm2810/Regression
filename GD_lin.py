import numpy as np

def eq(x, b0, b1):
    #Define the equation
    return b0+b1*x

def calculate_error(b0,b1,data):
    #Initialize error
    SSE = 0
    for i in range(0,len(data),1):
        x = data[i,0] #1st column in data is x
        y = data[i,1] #2nd column in data is y
        SSE += (y - eq(x,b0,b1))**2
    #Calculate MSE from SSE
    MSE = SSE/float(len(data))
    return MSE

def step_gradient(b0_current, b1_current, data, learning_rate):
    #initialize derivatives
    b0_deriv=0
    b1_deriv=0
    #Number of data points
    N = float(len(data))
    
    for i in range(0,len(data),1):
        x = data[i,0] #1st column in data is x
        y = data[i,1] #2nd column in data is y
        b0_deriv += -(2/N) * (y - ((b1_current * x) + b0_current))
        b1_deriv += -(2/N) * x * (y - ((b1_current * x) + b0_current))
        
    b0_new = b0_current - (learning_rate * b0_deriv)
    b1_new = b1_current - (learning_rate * b1_deriv)
        
    return b0_new,b1_new

def gradient_descent_runner(data, b0_start, b1_start, learning_rate, iter):    
    #Initialize weights
    b0 = b0_start
    b1 = b1_start
    #Run step gradient iteratively to update weights
    for i in range(iter):
        b0, b1 = step_gradient(b0, b1, np.array(data), learning_rate)     
    #Send weights to run function    
    return b0, b1
   
def run():    
    #import data
    data = np.genfromtxt('data-GD_lin_Siraj.csv', delimiter=',')
    #hyperparameters
    learning_rate = 0.0001
    b0_ini = 0
    b1_ini = 0
    iter = 10000
    #Print start
    print ('Starting gradient descent at b0 = {0}, b1 = {1}, MSE = {2}'.format(b0_ini, b1_ini, calculate_error(b0_ini, b1_ini, data)))
    print('Running....')
    #Run GD function
    b0, b1 = gradient_descent_runner(data, b0_ini, b1_ini, learning_rate, iter)
    #Print results
    print('After {0} iterations b0 = {1}, b1 = {2}, MSE = {3}'.format(iter, b0, b1, calculate_error(b0, b1, data)))

if __name__ == '__main__':
    run()
    

    
    