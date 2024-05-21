"""
Copyright c 2024 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""
#%%#Import the Functions
#Import the Functions
#********************#
from FunDefV39_2 import *
#********************#

#%% Import the csv file into a dataframe
filename = "Sil_t_DCMMeOH_496_50_0.csv"
csv = pd.read_csv(filename,header = None)
print('Imported file: ' + filename)

#%% Extract the time and amplitude information
DataX = np.array((csv.iloc[:,0]))
DataY = csv.iloc[:,1]
# DataY = np.log10(csv.iloc[:,1])

# Remove all the inf and nan values from the data
DataX,DataY = DataX[np.logical_not(np.isnan(DataX))], DataY[np.logical_not(np.isnan(DataX))] # removes rows containing NaN from the data
DataX,DataY = DataX[np.logical_not(np.isinf(DataX))], DataY[np.logical_not(np.isinf(DataX))] # removes rows containing inf from the data
DataX,DataY = DataX[np.logical_not(np.isinf(DataY))], DataY[np.logical_not(np.isinf(DataY))] # removes rows containing inf from the data

plt.plot(DataX, DataY)
plt.xlabel('DataX')
plt.ylabel('DataY')
plt.grid()
plt.title('Data as Imported')


#%% Interpolation: use if the dataset is too large
interpolateData = False
# interpolateData = True
if interpolateData:
    startInterpAfter = 100
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3.0))
    f = interp1d(DataX[startInterpAfter:], DataY[startInterpAfter:], kind='linear')
    xnew = np.linspace(DataX[startInterpAfter], DataX[-1], num=1000, endpoint=True)
    plt.plot(DataX, DataY, 'o', np.append(DataX[:startInterpAfter],xnew), np.append(DataY[:startInterpAfter],f(xnew)), 'x')
    DataX,DataY = np.append(DataX[:startInterpAfter],xnew),np.append(DataY[:startInterpAfter],f(xnew))
    plt.xscale('log')
    plt.show()

#%% Fitting settings and calls
# Biexponential function fixed baxkgrund in logtime
def BiexpFunFixBLog(t, U1, A1, U2, A2):#, B = 666):
    
    ret = A1*np.exp(-np.log(t)/np.log(U1)) + A2*np.exp(-np.log(t)/np.log(U2)) + 2.4
    
    return ret

# Biexponential function fixed baxkgrund in logtime
def BiexpFunFixBLogA(t, T1, A1, T2, A2):#, B = 666):
    #log(a)+log(b)=log(ab)
    #log(a+b)=???
    ret = np.log10(A1*(-(t)/(T1)) * A2*np.log10(-(t)/(T2)) + 2.4)
    return ret

def BiexpFunLog10Data(t, T1, A1, T2, A2):
    ret = (A1*(-(t)/(T1)) * A2*(-(t)/(T2)) + 2.4)
    return ret

# Choose which function to fit to
function = BiexpFunFixB

beoundRelax = 0.1
TBounds = (DataX[0] ,  DataX[-1])
ABounds = (min(DataY)*(1-beoundRelax), max(DataY)*(1+beoundRelax))
BBounds = (0, 1e3)

#Michael Sil50 (pdf) amplitude weighted PQ parameters
Mti = 1890
Mt1 = 13620
MA1 = 1.5395e+03
# MA1 = 1.34e3
Mt2 = 6050
MA2 = 1.0524e+04
# MA2 = 7.7e3
MB  = 2.4

Mcomp = [Mt1, MA1, Mt2, MA2, MB]

# Mcomp = [Mt1, np.log10(MA1), Mt2, np.log10(MA2), MB]
Mcomp = Mcomp[0:-1]
fullFitBack = 2.26428
BBounds = (0, 4)
breakpoint()
# funBounds = [TBounds, ABounds, TBounds, ABounds, BBounds] #Biexp fit with Bck
# funBounds = [np.log(TBounds), ABounds, np.log(TBounds), ABounds] #Biexp fit hard code backg
funBounds = [TBounds, ABounds, TBounds, ABounds]
# funBounds = [TBounds, np.log10(ABounds), TBounds, np.log10(ABounds)]
breakpoint()

# DataX,DataY = DataX[np.logical_not(np.isnan(np.log10(DataY)))], DataY[np.logical_not(np.isnan(np.log10(DataY)))] # removes rows containing NaN from the data


ret = heavyTailFit(x = DataX, y = DataY, fun = function, pBounds = funBounds, maxiterations = 1000)
cost = ret.fun



print("Data fit done!")
print("cost: " + str(cost))

#%% =================
fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))

xx_lin = np.linspace(0,DataX[-1],1000)

axs.scatter(DataX,DataY, color = 'black', label = 'data', marker = 'o', s = 5)
# axs.plot(xx_lin, function(xx_lin,*ret.x), color = 'blue')
plt.plot(DataX, function(DataX,*Mcomp), color = 'orange')
axs.plot(DataX, function(DataX,*ret.x))
# =======
axs.set_xlabel(r"$t\ [ns]$", **label_style)
axs.set_ylabel(r"Quantitiy", **label_style)
axs.grid(color='k', linestyle='-', linewidth=.05 , which ='both')

axs.set_yscale('log')

# %%
