"""
Copyright c 2024 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""

# Import the Functions
from FunDefV39_2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

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

# Remove all the inf and nan values from the data
valid_indices = np.logical_not(np.isnan(DataX) | np.isnan(DataY) | np.isinf(DataX) | np.isinf(DataY))
DataX, DataY = DataX[valid_indices], DataY[valid_indices]

plt.plot(DataX, DataY)
plt.xlabel('DataX')
plt.ylabel('DataY')
plt.grid()
plt.title('Data as Imported')
plt.show()

# Interpolation: use if the dataset is too large
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

# Fitting settings and calls
def BiexpFun(t, T1, A1, T2, A2, B):
    return A1 * np.exp(-t / T1) + A2 * np.exp(-t / T2) + B

# Adam optimization fit method
def adam_optimization_fit(x, y, initial_params, learning_rate=0.01, epochs=1000):
    params = tf.Variable(initial_params, dtype=tf.float32)

    def loss_fn():
        predictions = BiexpFun(x, params[0], params[1], params[2], params[3], params[4])
        return tf.reduce_mean(tf.square(predictions - y))

    optimizer = Adam(learning_rate)

    for epoch in range(epochs):
        optimizer.minimize(loss_fn, var_list=[params])
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss_fn().numpy()}')

    return params.numpy()

# Initial guess for the parameters
initial_params = [1, 1, 1, 1, 1]  # [T1, A1, T2, A2, B]

# Normalize DataX and DataY for better optimization
DataX_norm = DataX / np.max(DataX)
DataY_norm = DataY / np.max(DataY)

# Fit the data using Adam optimization
optimized_params = adam_optimization_fit(DataX_norm, DataY_norm, initial_params)

# Denormalize parameters if necessary
optimized_params[1] *= np.max(DataY)
optimized_params[3] *= np.max(DataY)

print("Optimized parameters using Adam:")
print(optimized_params)

# Plot the results
fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))

xx_lin = np.linspace(0, DataX[-1], 1000)
#%%
# axs.scatter(DataX, DataY, color='black', label='data', marker='o', s=5)
axs.plot(DataX, BiexpFun(DataX, *optimized_params), color='blue', label='Adam Fit')
axs.set_xlabel(r"$t\ [ns]$")
axs.set_ylabel(r"Quantity")
axs.grid(color='k', linestyle='-', linewidth=.05, which='both')
# axs.set_yscale('log')
plt.legend()
plt.show()

# %%
