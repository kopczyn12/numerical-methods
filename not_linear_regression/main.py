
import numpy as np
import math
from scipy.optimize import fmin
import matplotlib.pyplot as plt

#Loading data
file = "/home/mkopcz/Desktop/mn/lab6/data8.txt"
data = np.loadtxt(file)

#Time, value
t = np.array(np.copy(data[:, 0]))
y = np.copy(data[:, 1])


#Cost function
def cost_function(params):
    k, tau, zeta, tau_z = params

    omega_n = 1 / tau
    omega_zero = (math.sqrt(1 - (zeta ** 2))) / tau

    g_t = impulse_response_g_t(omega_n, zeta,omega_zero, t)
    h_t = step_response_h_t(omega_n, zeta,omega_zero, t)
    
    ys = np.mat(k * (tau_z * g_t + h_t))            

    y_ = np.mat(y)
    return ((ys - y_) * (ys - y_).T).item(0)        

# I 
def impulse_response_g_t(wn, zeta, w0, t):
    return (wn / np.sqrt(1 - zeta**2)) * np.multiply(np.exp( -zeta * wn * t), np.sin(w0 * t))     

# II 
def step_response_h_t(wn, zeta, w0, t):
    return 1 - np.multiply((np.exp( -zeta * wn * t)), np.cos(w0 * t)   + (zeta * np.sin(w0 * t) / np.sqrt(1 - zeta**2)))     


#jump
def step_response(params):      
    k, tau, zeta, tau_z = params

    omega_n = 1 / tau
    omega_zero = (math.sqrt(1 - (zeta ** 2))) / tau

    g_t = impulse_response_g_t(omega_n, zeta,omega_zero, t)
    h_t = step_response_h_t(omega_n, zeta,omega_zero, t)
    
    ys = k * (tau_z * g_t + h_t)   
    
    return  ys

initial_params = np.array([1.7, 2.5, 0.2, 10])
params_out = fmin(cost_function, initial_params)

# Plotting 

plt.plot(t,y , t , step_response(params_out))
plt.rcParams['figure.figsize'] = [18, 8]
plt.legend(["original", "approximation"])
plt.grid()
plt.show()