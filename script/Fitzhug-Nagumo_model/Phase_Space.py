import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/.../Local_Phase_Space'

# this will creat the path if it does not exist already 
if not os.path.isdir (path):
    os.mkdir(path)

W_Poly = np.load('/home/.../Nullclines/W_Poly.npy')
W_Linear = np.load('/home/.../W_Linear.npy')
V_values_Nullclines = np.load('/home/.../Nullclines/V_values.npy')




# Params
a = 3  # excitation for the threshold
b = -0.05 
epsilon = 0.01

# initialization of dynamical varibales 
V = 0
W = 0.1

V_values = []
W_values = []


# time-related params/variables
dt = 0.01         # time step
nt = np.power(10, 6)       # total time steps
time = nt*dt

# stimulation current
I = 0
I_values = []
for i in range (0, nt):


    # angebraic equation: Polynomial equation of third degree
    f_v = a*V*(V - b)*(1 - V)

    # ODE equations solved by Euler method
    V_new = (f_v - W )*dt + I + V 
    W_new = (epsilon*(V - W))*dt + W
    

    # adding the recent value of each varibles to the correponding list
    V_values.append(V_new)
    W_values.append(W_new)

    # initialising the new value for each variables
    V = V_new
    W = W_new
plt.axes().set_aspect(1)
plt.plot(V_values, W_values, 'b', linewidth = 1.5)
plt.plot(V_values_Nullclines, W_Poly, 'k-', linewidth = 0.5)
plt.plot(V_values_Nullclines, W_Linear, 'r-', linewidth = 0.5)
plt.ylabel('W (a.u.)', size = 15)
plt.xlabel('V (a.u.)', size = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig(os.path.join(path, 'Phase_Space.png'))
plt.savefig(os.path.join(path, 'Phase_Space.svg'))
plt.show()
