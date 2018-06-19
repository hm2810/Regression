import numpy as np

#Import data
data = np.genfromtxt('data-GD_lin_Siraj.csv', delimiter=',')
#hyperparameters
learning_rate = 0.0001
iter = 1000

# =============================================================================
# Loss function is Error**2, which is y_actual-y_predicted
#  Now for gradient descent, we need to find derivative of Loss function wrt b0 and b1
# For b0, from chain rule:
#  d(Error**2)/d(b0) = d(Error**2)/d(Error) * d(Error)/d(b0)
#  = 2 * Error * d(Error)/d(b0)
#  = 2 * (y_actual - b0+b1*x) * d(y_actual - (b0+b1*x))/b0
#  = 2 * (y_actual - b0+b1*x) * (0 - 1 - 0)
#  = -2 * (y_actual - y_predicted)
# Similarly, for b1, d(Error**2)/d(b1) = 2 * (y_actual - b0+b1*x) * (0 - 0 - x)
#  = -2 * (y_actual - y_predicted) * x
# Now, the b1 will be changed as, b1_new = b1_old - (deriv_b1 * learning rate)
# Note that we are subtracting the derivative because derivative is negative,
# however, if we have defined error as (y_predicted - y_actual) then derivative
# would have been +ve so we had to add the derivative in above eq. instead of subtracting. 
# 
# =============================================================================

def eq(x, b0, b1):
    #Define the equation
    return b0+b1*x

def calculate_error(b0,b1,data):
    #Initialize error
    SSE = 0
    #Iteration over data list
    for i in range(0,len(data),1):
        x = data[i,0] #1st column in data is x
        y = data[i,1] #2nd column in data is y
        SSE += (y - eq(x,b0,b1))**2
    #Calculate MSE from SSE
    MSE = SSE/float(len(data))
    return MSE




def GDrun():    
    #weights initialize
    b0 = 0
    b1 = 0
    #Number of data points
    N = float(len(data))
    #Print start
    print ('Starting gradient descent at b0 = {0}, b1 = {1}, MSE = {2}'.format(b0, b1, calculate_error(b0, b1, data)))
    print('Running....')

    #Run step gradient iteratively to update weights (Iteration over time-step)
    for i in range(iter):
        b0_deriv = 0
        b1_deriv = 0
        #Iteration over data list
        for i in range(0,len(data),1):
            x = data[i,0] #1st column in data is x
            y = data[i,1] #2nd column in data is y
            b0_deriv += -(2/N) * (y - eq(x,b0,b1)) #Explained above
            b1_deriv += -(2/N) * x * (y - eq(x,b0,b1)) #Explained above
        b0 = b0 - (b0_deriv * learning_rate) #Explained above
        b1 = b1 - (b1_deriv * learning_rate) #Explained above
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
    


    
    