import numpy as np 


from scipy.signal import hilbert

import os, sys
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)


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
output_file = f'/scratch01.local/ipellini/V_values_brain/V_values_m={block}_seed={seed}.pkl'
simulation_data = read_simulation_data(output_file)
simulation_data = simulation_data.T
if simulation_data is not None:
    print(simulation_data.shape)

# Function to compute the phase of each element across time using the Hilbert transform

def compute_phases(data):
    """
    Compute the phase of each element using the Hilbert transform.
    
    Parameters:
        data (np.ndarray): An (N, T) array where N is the number of elements and T is the time dimension.
        
    Returns:
        np.ndarray: An (N, T) array containing the phases.
    """
    # Apply Hilbert transform along the time axis
    analytic_signal = hilbert(data, axis=1)
    # Extract the phase of the analytic signal
    phases = np.angle(analytic_signal)
    return phases


def kuramoto_order_parameter(phases):
    """
    Compute the Kuramoto order parameter for the given phases.
    
    Parameters:
        phases (jnp.ndarray): An (N, T) array of phases for N elements over T time points.
        
    Returns:
        tuple:
            - jnp.ndarray: A (T,) array representing the amplitude of the Kuramoto order parameter over time.
            - jnp.ndarray: A (T,) array representing the phase of the Kuramoto order parameter over time.
    """
    N = phases.shape[0]
    
    order_parameter_complex = jnp.sum(jnp.exp(1j * phases), axis=0) / N
    
    amplitude = jnp.mean(jnp.abs(order_parameter_complex))
   
    #phase = jnp.angle(order_parameter_complex)
    return amplitude
output_file = f'/scratch01.local/ipellini/Kuramoto_brain.pkl'

# Convert R to a float

simulation_data=compute_phases(simulation_data)

R=kuramoto_order_parameter(simulation_data)
R = float(R)
# Prepare the data to be dumped
data_to_dump = {
    'seed': seed,
    'block': block,
    'R': R
}

# Write the header if the file does not exist
if not os.path.exists(output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(['seed', 'm', 'R'], f)

# Append the data to the file
with open(output_file, 'ab') as f:
    pickle.dump(data_to_dump, f)

