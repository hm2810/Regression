import numpy as np

#Import data
data = np.genfromtxt('data-GD_lin_Siraj.csv', delimiter=',')
#hyperparameters
learning_rate = 0.0001
iter = 10


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


def GDrun():    
    #weights initialize
    b0 = 0
    b1 = 0
    #Print start
    print ('Starting gradient descent at b0 = {0}, b1 = {1}, MSE = {2}'.format(b0, b1, calculate_error(b0, b1, data)))
    print('Running....')
    #Run GD function
    #Run step gradient iteratively to update weights
    for i in range(iter):
        b0, b1 = step_gradient(b0, b1, data, learning_rate)
        #Print results
        print('After {0} iterations b0 = {1}, b1 = {2}, MSE = {3}'.format(i, b0, b1, calculate_error(b0, b1, data)))
    print('End....')
    
    #Turn fitted parameters to global parameters
    global fitted_para
    fitted_para = [b0,b1]

#Plotting
def plot():
    import matplotlib.pyplot as pt
    import math as mt
    x = data[:,0]
    y = data[:,1]
    yfit = [eq(j,*fitted_para) for j in x] 
    pt.plot(x,y,'o')
    pt.plot(x,yfit,'r--',label='Fit')
    pt.legend(loc='upper left')
    #Calculating statistical indicators
    SSE_yfit = sum((yfit - y)**2)
    SSR_yfit = sum((yfit - np.mean(y))**2)
    SST_yfit = SSE_yfit + SSR_yfit 
    Rsq_yfit = SSR_yfit/SST_yfit  
    RMSE_yfit = mt.sqrt(SSE_yfit/len(yfit))
    print('R2_yfit= ',Rsq_yfit,'RMSE_yfit= ',RMSE_yfit)
 
if __name__ == '__main__':
    GDrun()
    plot()
    


    
    