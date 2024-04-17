import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/shussai/Documents/Postdoc/15Jan2024_Biophysics_Course/simulation_data/Single_Cell'

# this will creat the path if it does not exist already 
if not os.path.isdir (path):
    os.mkdir(path)

# Params
a = 3  
b = 0.02 
epsilon = 0.01

# initialization of dynamical varibales 
V = 0
W = 0

# defining a list for the values of the dynamical variables
# each list is initialised with the initial value of each dynamical variables
V_values = []
W_values = []


# time-related params/variables
dt = 0.01         # time step
nt = np.power(10, 5)       # total time steps
time = nt*dt



# stimulation current
I = 0
I_values = []
counter = 0
for i in range (0, nt):
    
    if 0 < i < 6:
        I = 0.006
        I_values.append(I)

    elif 20000 < i < 20000 + 6:
        I = 0.02
        I_values.append(I)

    elif 32300 < i< 32300 + 6:
        I = 0.02
        I_values.append(I)

    else:
        I = 0
        I_values.append(I)

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

#plotting
#plt.axes().set_aspect(6000)
plt.plot(np.linspace(0, time, nt), V_values, 'k', linewidth = 1.5, label = 'Action potential')
plt.plot(np.linspace(0, time, nt), I_values, 'r', linewidth = 1, label = 'External current')
plt.ylabel('V (a.u.)', size = 15)
plt.xlabel('t (a.u.)', size = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.tight_layout()
plt.ylim(-0.4, 1.1)
plt.savefig(os.path.join(path, 'V_external_train_stimuli.png'))
plt.savefig(os.path.join(path, 'V_external_train_stimuli.svg'))
plt.show()

