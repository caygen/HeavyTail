"""
Copyright c 2024 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""
#%%
# Import the Functions
from FunDefV39_2 import *
# from FunDefV39_2_minimize import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#%%
# Import the csv file into a dataframe
filename = "Sil_t_DCMMeOH_496_50_0.csv"
try:
    csv = pd.read_csv(filename, header=None)
    print(f'Imported file: {filename}')
except FileNotFoundError:
    print(f"File {filename} not found.")
    exit()

# Extract the time and amplitude information
DataX = np.array(csv.iloc[:, 0])
DataY = csv.iloc[:, 1]
#%%
# Remove all the inf and nan values from the data
valid_indices = np.logical_not(np.isnan(DataX) | np.isnan(DataY) | np.isinf(DataX) | np.isinf(DataY))
DataX, DataY = DataX[valid_indices], DataY[valid_indices]

plt.plot(DataX, DataY)
plt.xlabel('DataX')
plt.ylabel('DataY')
plt.grid()
plt.title('Data as Imported')
plt.show()

#%% Interpolation: use if the dataset is too large
interpolateData = False
if interpolateData:
    startInterpAfter = 100
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3.0))
    f = interp1d(DataX[startInterpAfter:], DataY[startInterpAfter:], kind='linear')
    xnew = np.linspace(DataX[startInterpAfter], DataX[-1], num=1000, endpoint=True)
    plt.plot(DataX, DataY, 'o', np.append(DataX[:startInterpAfter], xnew), np.append(DataY[:startInterpAfter], f(xnew)), 'x')
    DataX, DataY = np.append(DataX[:startInterpAfter], xnew), np.append(DataY[:startInterpAfter], f(xnew))
    plt.xscale('log')
    plt.show()

#%% Fitting settings and calls
def BiexpFunFixBLog(t, U1, A1, U2, A2):
    return A1 * np.exp(-np.log(t) / np.log(U1)) + A2 * np.exp(-np.log(t) / np.log(U2)) + 2.4

def BiexpFunFixBLogA(t, T1, A1, T2, A2):
    return np.log10(A1 * (-(t) / (T1)) * A2 * np.log10(-(t) / (T2)) + 2.4)

def BiexpFunLog10Data(t, T1, A1, T2, A2):
    return A1 * (-(t) / (T1)) * A2 * (-(t) / (T2)) + 2.4

#%% Choose which function to fit to
function = BiexpFunFixB

beoundRelax = 0.1
TBounds = (DataX[0], DataX[-1])
ABounds = (min(DataY) * (1 - beoundRelax), max(DataY) * (1 + beoundRelax))

# Michael Sil50 (pdf) amplitude weighted PQ parameters
Mti = 1890
Mt1 = 13620
MA1 = 1.5395e+03
Mt2 = 6050
MA2 = 1.0524e+04
MB = 2.4

Mcomp = [Mt1, MA1, Mt2, MA2, MB]
Mcomp = Mcomp[:-1]
funBounds = [TBounds, ABounds, TBounds, ABounds]
#%%
ret = heavyTailFit(x=DataX, y=DataY, fun=function, pBounds=funBounds, maxiterations=1000)
cost = ret.fun

print("Data fit done!")
print(f"SA cost: {cost}")
Can_cost = ChiSquaredCost(DataX, DataY, function, ret.x)
print(f"Can cost: {Can_cost}")

PQ_cost = ChiSquaredCost(DataX, DataY, function, Mcomp)
print(f"PQ cost: {PQ_cost}")

##
# Plot the results
fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))

xx_lin = np.linspace(0, DataX[-1], 1000)

axs.scatter(DataX, DataY, color='black', label='data', marker='o', s=5)
axs.plot(DataX, function(DataX, *Mcomp), color='orange')
axs.plot(DataX, function(DataX, *ret.x))

axs.set_xlabel(r"$t\ [ns]$")
axs.set_ylabel(r"Quantity")
axs.grid(color='k', linestyle='-', linewidth=.05, which='both')
axs.set_yscale('log')

plt.show()
##%

# %%
