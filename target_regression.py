import numpy as np
import scipy as sc
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm




###defining your fitfunction
def eq(x, a, b, c):
    return a*x**2+b*x+c
 
###function end

#Training data
xt = np.array([1, 2, 3, 4, 5])
yt = np.array([0.5,0.2,3.7,4.4,6.5])

#Make initial guess for fitting parameters and do guess fitting
ig=[1, 1,-.01]
gf=[eq(x,*ig) for x in xt] #*ig behaves simular to ig[0],ig[1],ig[2]

#making the actual fit
popt,pcov = curve_fit(eq, xt, yt,ig)

###preparing unknown data for showing the fit
xu = xt  #Use known data, not xt 
#xu = np.linspace(min(xt),max(xt),len(xt))
yu=[eq(x, *popt) for x in xu]
yk = yt #yk should be known, to calculate Rsq and validate the model

#plot
plt.plot(xt,yt,linestyle='',marker='o', color='r',label="data")
plt.plot(xt,gf,linestyle='',marker='^', color='b',label="initial guess")
plt.plot(xu,yu,linestyle='--', color='#900000',label="fit with ({0:0.2g},{1:0.2g},{2:0.2g})".format(*popt))

plt.legend(loc=0, title="graphs", fontsize=12)

#calculate rsq
err_y = yu - yk
SSerr_y = sum(err_y**2)
SStotal_y = len(yk) * sc.var(yk)
Rsq_y = 1 - (SSerr_y/SStotal_y)


#import numpy as np
#import scipy as sc
#import matplotlib.pyplot as mp
#
#
#x = np.array([1,2,3,4,5])
#y = np.array([0.5,0.2,3.7,4.4,6.5])
#
##curve fitting
#f1 = sc.polyfit(x,y,1) # 1-order
#f2 = sc.polyfit(x,y,2) # 2-order
#f3 = sc.polyfit(x,y,3) # 3-order
#
##plot actual data with known x
#mp.plot(x,y,'o') 
#
##unknown x (range from 1 to 5, with 5 points)
#xu = sc.linspace(1,5,5)  
#
##determine new y corresponding unknown x using fitted parameters above
#y_f1 = sc.polyval(f1,xu)   # or y_f1 = f1[0]*xu+f1[1]
#y_f2 = sc.polyval(f2,xu)   # or y_f2 = f2[0]*xu**2+f2[1]*xu+f2[2]
#y_f3 = sc.polyval(f3,xu)   # or y_f3 = f3[0]*xu**3+f3[1]*xu**2+f3[2]*xu+f3[3]
#
##plot new x corresponding to new y  
#mp.plot(xu,y_f1,'r--')
#mp.plot(xu,y_f2,'b--')
#mp.plot(xu,y_f3,'y--')
#
##Rsq for unknown x
#err_y_f1 = y_f1 - y
#SSerr_y_f1 = sum(err_y_f1**2)
#SStotal_y_f1 = len(y) * sc.var(y)
#Rsq_y_f1 = 1 - (SSerr_y_f1/SStotal_y_f1)
#
#err_y_f2 = y_f2 - y
#SSerr_y_f2 = sum(err_y_f2**2)
#SStotal_y_f2 = len(y) * sc.var(y)
#Rsq_y_f2 = 1 - (SSerr_y_f2/SStotal_y_f2)
#
#err_y_f3 = y_f3 - y
#SSerr_y_f3 = sum(err_y_f3**2)
#SStotal_y_f3 = len(y) * sc.var(y)
#Rsq_y_f3 = 1 - (SSerr_y_f3/SStotal_y_f3)
#
##Alternate builtin linregress function: [`] = sc.stats.linregress(x,y) but gives 'R' in matrix not 'Rsq'
##but, for non-lin regress a target equation is needed for curve fitting
#
#
