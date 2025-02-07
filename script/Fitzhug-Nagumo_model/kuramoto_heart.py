import numpy as np 


from scipy.signal import hilbert

import os, sys
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)

from jax import lax
import pickle
import jax.numpy as jnp

import argparse

parser = argparse.ArgumentParser(description='Run simulation with specified seed and block.')
parser.add_argument('--seed', type=int, required=True, help='Random seed for simulation.')
parser.add_argument('--block', type=float, required=True, help='Conduction block threshold.')
args = parser.parse_args()

seed = args.seed
block = args.block
print(f"Seed: {seed}, Block: {block}")

def read_simulation_data(file_path):
    """
    Reads the simulation data from a pickle file and returns it as a JAX array.
    
    Parameters:
    file_path (str): The path to the pickle file containing the simulation data.
    
    Returns:
    jnp.ndarray: The simulation data as a JAX array.
    """
    data = []
    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    vs = pickle.load(f)
                    data.append(vs)
                except EOFError:
                    break
        return jnp.concatenate(data, axis=0)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

# Example usage
output_file = f'V_values_p={block}_seed={seed}.pkl'
simulation_data = read_simulation_data(output_file)
simulation_data = simulation_data.T
if simulation_data is not None:
    print(simulation_data.shape)

def compute_phases(data):
    """
    Compute the phase of each element using the Hilbert transform.
    
    Parameters:
        data (jnp.ndarray): An (N, T) array where N is the number of elements and T is the time dimension.
        
    Returns:
        jnp.ndarray: An (N, T) array containing the phases.
    """
    # Apply Hilbert transform along the time axis
    analytic_signal = hilbert(data, axis=1)
    # Extract the phase of the analytic signal
    phases = jnp.angle(analytic_signal)
    return phases


def kuramoto_order_parameter(phases):
    """
    Compute the Kuramoto order parameter for the given phases.
    
    Parameters:
        phases (jnp.ndarray): An (N, T) array of phases for N elements over T time points.
        
    Returns:
        tuple:
            - jnp.ndarray: A (T,) array representing the amplitude of the Kuramoto order parameter over time
            - jnp.ndarray: A (T,) array representing the phase of the Kuramoto order parameter over time.
    """
    N = phases.shape[0]
    
    order_parameter_complex = jnp.sum(jnp.exp(1j * phases), axis=0) / N
    
    amplitude = jnp.abs(order_parameter_complex)
   
    phase = jnp.angle(order_parameter_complex)
    return amplitude, phase

#here I could again go down to 4 from 15 for the noise
def tot_kuramoto_heart(u_sol, c1, N_x,N_y):
    """
    Calculate the Kuramoto order parameter for a given solution matrix and mask.
    Parameters:
    u_sol (numpy.ndarray): The solution matrix with shape (timesteps, N_x * N_y).
    c1 (numpy.ndarray): The mask matrix with shape (N_x, N_y).
    N_x (int): The number of grid points in the x-direction.
    N_y (int): The number of grid points in the y-direction.
    Returns:
    float: The Kuramoto order parameter.
    """

    #u_sol=u_sol.T
    u_sol=u_sol.reshape(N_x,N_y,-1)

    c1=c1.reshape(N_x,N_y)
    cnot=~c1[4:(N_x-4), 4:(N_x-4)]  #remove borders to avoid boundary effects
   

    usol_phase=u_sol[4:(N_x-4),4:(N_x-4),:]

    usol_phase=usol_phase.reshape((N_x-8)*(N_y-8),-1)
    
    phases=compute_phases(usol_phase[:, :])
    phases=phases[:,:]  #here to be changes to cover from equilibration to a bunch of stuff before e
    u_sol_phase=phases.reshape(N_x-8,N_y-8,-1)

    R=[]
    
    psi=[]
    for j in range(N_x-8):  # Iterate over the correct dimension
    # Apply the mask to filter out the elements that are False in cnot
        filtered_column = u_sol_phase[:, j, :][cnot[:, j]]
    
        r1, psi1=kuramoto_order_parameter(filtered_column)
        R.append(r1)
        psi.append(psi1)
    R=jnp.array(R)
   
    return jnp.mean(R)
def generate_laplacian(N, M, conduction_block_threshold, seed=0):

    # Generate random conduction blocks
    np.random.seed(seed)
    conduction_blocks = np.random.rand(N, M) < conduction_block_threshold

    return jnp.array(conduction_blocks)

c1=generate_laplacian(200, 200, block, seed)

R=tot_kuramoto_heart(simulation_data, c1, 200, 200)

output_file = f'Kuramoto_heart.pkl'
data_to_dump = {
    'seed': seed,
    'block': block,
    'R': R
}

# Write the header if the file does not exist
if not os.path.exists(output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(['seed', 'p', 'R'], f)

# Append the data to the file
with open(output_file, 'ab') as f:
    pickle.dump(data_to_dump, f)

