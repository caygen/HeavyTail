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
from scipy.integrate import cumulative_simpson
from scipy.constants import elementary_charge
#%%
# Import the csv file into a dataframe
# filename = "Sil_t_DCMMeOH_496_50_0.csv"
# rawfile  = "Sil_t_DCMMeOH_496_50_0_full.csv"
# filename = "May2_AlGaN_LED_ON 2.csv"
# filename = "May8_AlGaN_LED_OFF.csv"
filename = "012_M81_ETH_F240417A_L2_n_WT_new_200K_10MohmLED_0u1A_ON.csv"

try:
    csv = pd.read_csv(filename)
    print(f'Imported file: {filename}')
except FileNotFoundError:
    print(f"File {filename} not found.")
    exit()

#%%
# raw = pd.read_csv(rawfile, delimiter='\t')
# Extract the time and amplitude information
# DataX = np.array(csv.iloc[:-1, 0])
# DataY = cumulative_simpson(np.array(csv.iloc[:, 1]))/1e4
current = 10e-6
DataX   = np.log(np.array(csv['Elapsed Time (s)']))
Rxy_X	= np.array(csv['Rxy_X'])/current #ohms
Ryx_X   = np.array(csv['Ryx_X'])/current #ohms
Bfield  = np.array(csv['Field (T)'])
R_H     = (Rxy_X-Ryx_X)/(2*Bfield)
n2d     = 1/(R_H * elementary_charge *1e4) * 1e-13 #10^13 cm^-2
DataY = n2d
plt.plot(DataX, DataY)
plt.grid()
# plt.xscale('log')

#%%
# Remove all the inf and nan values from the data
valid_indices = np.logical_not(np.isnan(DataX) | np.isnan(DataY) | np.isinf(DataX) | np.isinf(DataY))
DataX, DataY = DataX[valid_indices], DataY[valid_indices]
# interpolate the data
newX = np.linspace(DataX[0], DataX[-1], 1000)
newY = interp1d(DataX, DataY, kind='cubic')(newX)
# DataX = new



plt.plot(DataX, DataY)
plt.plot(newX, newY)
plt.xlabel('DataX')
plt.ylabel('DataY')
plt.grid()
plt.title('Data as Imported')
# plt.xscale('log')
plt.show()


#%% Choose which function to fit to
# function = BiexpFun
# function = BiexpFunFixB
def FHTx(x,u,B,df0,m):
#     if m == 0: #if if it at the SE limit
#         return df0*np.exp(-np.exp(B*(x-u)))
#     if m == 1: #if HT is at the AD limit
#         return ADx(x,u,B,df0)
    return df0*(1-m)/(np.exp((1-m)*np.exp(B*(x-u)))-m)

function = FHTx


#%% FIT HERE
# FIT HERE
funBounds = [(0,30), (0,1), (-20,20),(0.999,1)] # illumination
# funBounds = [(0,12), (0,1), (-20,20),(0,1)] # dark

# ret = heavyTailFit(DataX, DataY, function, pBounds=funBounds)
ret = dual_annealing(residualModSig, bounds=funBounds, args=(newX, newY, function))
# optimized = curve_fit(function, DataX, DataY)
# optimized2 = curve_fit(function, DataX, DataY, method=)


#%% Offset Calc
FdataAv    = OffsetAverage(DataX,DataY)
FfitAv     = OffsetAverage(newX,function(newX,*ret.x))
offset     = (FdataAv-FfitAv)

plt.plot(newX, newY)
plt.plot(newX, function(newX, *ret.x)+offset)
plt.title('Illumination')
# plt.title('Dark')
#%%
#plot linear time
plt.plot(np.exp(newX), newY)
plt.plot(np.exp(newX), function(newX, *ret.x)+offset)
plt.title('Illumination')
# plt.title('Dark')
plt.xscale('log')
# cost = ret.fun
# cost = ret.fun
#%%
print("Data fit done!")
print(f"SA cost: {cost}")
# Can_cost = ChiSquaredCost(DataX, DataY, function, ret.x)
# print(f"Can cost: {Can_cost}")

# PQ_cost = ChiSquaredCost(DataX, DataYIntegraded, function, Mcomp)
# print(f"PQ cost: {PQ_cost}")

##
# Plot the results
fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))

xx_lin = np.linspace(0, DataX[-1], 1000)

# axs.scatter(DataX, DataYIntegraded/1e6, color='black', label='data', marker='o', s=5)
# axs.plot(DataX, function(DataX, *Mcomp), color='orange')
axs.plot(DataX, function(DataX, *ret.x))

axs.set_xlabel(r"$t\ [ns]$")
axs.set_ylabel(r"Quantity")
axs.grid(color='k', linestyle='-', linewidth=.05, which='both')
axs.set_yscale('log')

plt.show()
##%

# %%
