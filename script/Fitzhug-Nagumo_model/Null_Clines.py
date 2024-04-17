import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/.../Nullclines'

# this will creat the path if it does not exist already 
if not os.path.isdir (path):
    os.mkdir(path)

# Params
a = 3  
b = -0.05 
epsilon = 0.01

# initialization of dynamical varibales 
V = 0
W = 0

# defining a list for the values of the dynamical variables
W_Poly = []
W_Linear = []

# external current
I = 0

V_values = list(np.arange(-0.25, 1.1, 0.01))
for i in range(0, len(V_values)):

    W_P = a*V_values[i]*(V_values[i] - b)*(1 - V_values[i]) + I

    W_L = V_values[i]

    # adding the recent value of each varibles to the correponding list
    W_Poly.append(W_P)
    W_Linear.append(W_L)

plt.axes().set_aspect(1)
plt.plot(V_values, W_Poly, 'k.', linewidth = 3)
plt.plot(V_values, W_Linear, 'r.', linewidth = 3)
plt.xlabel('V (a.u.)', size = 15)
plt.ylabel('W (a.u.)', size = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)


plt.show()

np.save(os.path.join(path, 'W_Poly.npy'), W_Poly)
np.save(os.path.join(path, 'W_Linear.npy'), W_Linear)
np.save(os.path.join(path, 'V_values.npy'), V_values)