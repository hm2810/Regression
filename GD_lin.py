import numpy as np
import matplotlib.pyplot as pt
import math  as mt

#Random
#x = sorted(np.random.randint(low=1,high=50, size=(10)),key=int)
x = np.array([11,12,13,14,15,16,17,18,19,20])
y = np.random.randint(low=101,high=200, size=(len(x)))
#y = np.array([111,112,123,144,175,156,197,218,219,220])

b0=0
b1=0
ygd = np.zeros(shape=(len(y)),dtype=np.int32)


def compute_error():
    error = np.zeros(shape=(len(y)),dtype=np.int32)
    
   
for i in range(0,len(x),1): #start(i) = 0, end(i) = len(x), interval = 1
    
    ygd[i] = b0 + b1*x[i] 
    error[i] = y[i] - ygd[i]
    b0= b0 + 
    b1= b1 + 
    print('Iteration:', i, str(', ygd')  , ygd, str(', error=')  , error)






# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))