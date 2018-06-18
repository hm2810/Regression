import numpy as np
import scipy as sc
import matplotlib.pyplot as mp
from matplotlib import rcParams

x = np.array([1,2,3,4,5])
y = np.array([0.5,0.2,3.7,4.4,6.5])

#curve fitting
f1 = sc.polyfit(x,y,1) # 1-order
f2 = sc.polyfit(x,y,2) # 2-order
f3 = sc.polyfit(x,y,3) # 3-order

#plot actual data with known x
mp.plot(x,y,'o') 

#unknown x (range from 1 to 5, with 5 points)
xu = x    #Preferably should be known for validation
#xu = sc.linspace(1,5,5)  

#determine new y corresponding unknown x using fitted parameters above
y_f1 = sc.polyval(f1,xu)   # or y_f1 = f1[0]*xu+f1[1]
y_f2 = sc.polyval(f2,xu)   # or y_f2 = f2[0]*xu**2+f2[1]*xu+f2[2]
y_f3 = sc.polyval(f3,xu)   # or y_f3 = f3[0]*xu**3+f3[1]*xu**2+f3[2]*xu+f3[3]

#plot new x corresponding to new y  
mp.plot(xu,y_f1,'r--',label='Linear')
mp.plot(xu,y_f2,'b--',label='Quad')
mp.plot(xu,y_f3,'y--',label='Cubic')
mp.xlabel('$x$')
mp.ylabel('$y$')
mp.legend(loc='upper left')
rcParams['figure.figsize'] = (10,5)
rcParams['legend.fontsize'] = 16
rcParams['axes.labelsize'] = 16

#Rsq for unknown x
err_y_f1 = y_f1 - y
SSerr_y_f1 = sum(err_y_f1**2)
SStotal_y_f1 = len(y) * sc.var(y)
Rsq_y_f1 = 1 - (SSerr_y_f1/SStotal_y_f1)

err_y_f2 = y_f2 - y
SSerr_y_f2 = sum(err_y_f2**2)
SStotal_y_f2 = len(y) * sc.var(y)
Rsq_y_f2 = 1 - (SSerr_y_f2/SStotal_y_f2)

err_y_f3 = y_f3 - y
SSerr_y_f3 = sum(err_y_f3**2)
SStotal_y_f3 = len(y) * sc.var(y)
Rsq_y_f3 = 1 - (SSerr_y_f3/SStotal_y_f3)

#Alternate builtin linregress function: [`] = sc.stats.linregress(x,y) but gives 'R' in matrix not 'Rsq'
#but, for non-lin regress a target equation is needed for curve fitting


