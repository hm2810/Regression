import numpy as np
import matplotlib.pyplot as pt
import math  as mt

#Random
#x = sorted(np.random.randint(low=1,high=50, size=(10)),key=int)
x = np.array([11,12,13,14,15,16,17,18,19,20])
y = np.random.randint(low=101,high=200, size=(len(x)))
#y = np.array([111,112,123,144,175,156,197,218,219,220])

b1_num=0
b1_den=0
    
for i in range(0,len(x),1): #start(i) = 0, end(i) = len(x), interval = 1
    
    b1_num = ((x[i]-np.mean(x)) * (y[i]-np.mean(y))) + b1_num
    b1_den = (x[i]-np.mean(x))**2 + b1_den
    print('Iteration:', i, str(', b1_num=')  , b1_num, str(', b1_den=')  , b1_den)

#Fitting analytical parameters
b1 = b1_num/b1_den
b0 = np.mean(y) - b1*np.mean(x)   
print('b0=',b0,', b1=',b1)
ycal = b0+b1*x

#Calculating statistical indicators
SSE_ycal = sum((ycal - y)**2)
SSR_ycal = sum((ycal - np.mean(y))**2)
SST_ycal = SSE_ycal + SSR_ycal #or SST_ycal = len(y) * sc.var(y)
Rsq_ycal = SSR_ycal/SST_ycal  #or Rsq_ycal = 1 - (SSE_ycal/SST_ycal)
RMSE_ycal = mt.sqrt(SSE_ycal/len(ycal))
print('R2_ycal= ',Rsq_ycal,'RMSE_ycal= ',RMSE_ycal)

#Plotting
pt.plot(x,y,'o')
pt.plot(x,ycal,'r--',label='Anafit')
pt.legend(loc='upper left')


#------------------For comparison with SciPy library-----------------------

import scipy as sc

#Fitting polyfit calculated paramters for comparison
fit = sc.polyfit(x,y,1)
b1_fit = fit[0]
b0_fit = fit[1]
yfit =  b0_fit+b1_fit*x
#yfit2 = sc.polyval(fit,x) #Easy alternative of yfit
print('b0_fit=',b0_fit,', b1_fit=',b1_fit)

#Calculating statistical indicators
SSE_yfit = sum((yfit - y)**2)
SSR_yfit = sum((yfit - np.mean(y))**2)
SST_yfit = SSE_yfit + SSR_yfit #or SST_yfit = len(y) * sc.var(y)
Rsq_yfit = SSR_yfit/SST_yfit  #or Rsq_yfit = 1 - (SSE_yfit/SST_yfit)
RMSE_yfit = mt.sqrt(SSE_yfit/len(yfit))
print('R2_yfit= ',Rsq_yfit,'RMSE_yfit= ',RMSE_yfit)

#plot
pt.plot(x,yfit,'b--',label='Polyfit')
pt.legend(loc='upper left')

    

    