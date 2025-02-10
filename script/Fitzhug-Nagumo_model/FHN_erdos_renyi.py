import numpy as np 

import gc
import pickle
import os, sys
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)



from scipy.sparse import diags, coo_matrix
from scipy import sparse
import jax.numpy as jnp
from jax import lax, jit
import jax.random as random
from jax.experimental import sparse
import argparse
import jax
import psutil
import networkx as nx
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

def log_memory_usage():

    process = psutil.Process()

    mem_info = process.memory_info()

    print(f"RSS: {mem_info.rss / 1024**2:.2f} MB, VMS: {mem_info.vms / 1024**2:.2f} MB")



parser = argparse.ArgumentParser(description='Run simulation with specified seed and block.')
parser.add_argument('--seed', type=int, required=True, help='Random seed for simulation.')
parser.add_argument('--block', type=float, required=True, help='Conduction block threshold.')
args = parser.parse_args()

seed = args.seed
m = args.block
print(f"Seed: {seed}, Block: {m}")



N_x = 200
N_y = 200
N = N_x * N_y
k=8

#random network implementation
def random_graph(N, k, J, seed=100, weights='homogeneous', weight_generator=None):
    '''
    Creates a random, directed, and weighted Erdos Renyi graph.
    Parameters:
        N: number of nodes
        k: mean nodal degree
        J: weight parameters. If homogeneous weights: constant float, if gaussian weigts: J=(J_mean, J_sigma)
        seed: seed for the ER graph generation
        weights: Type of weights, 'homogeneous' or 'gaussian'
        generator: random generator for random weights
    Returns:
        sparse jax.experimental coupling matrix 
    '''
    p = k / (N - 1)
    
    # Create ER graph
    G = nx.erdos_renyi_graph(N, p, directed=True, seed=seed)
    
    # Put weights
    for u, v in G.edges():
        if weights=='homogeneous':
            G[u][v]['weight'] = J#generator.normal(J, J/5)
        elif weights=='gaussian':
            G[u][v]['weight'] = weight_generator.normal(J[0], J[1])
    
    # Get the adjacency matrix in sparse format
    adj_matrix = nx.adjacency_matrix(G, weight='weight')
    
    return sparse.BCSR.from_scipy_sparse(adj_matrix)

def FHN_step(u, v, N, a, b, e, Du, sigma, L, key, delta_t):

    # Generate Gaussian noise for each element of h
    noise = random.normal(key, u.shape)
    
    du = a * u * (u - b) * (1 - u) + Du * (L @ u) - v 
    dv = e * (u - v)
    u_new = u + du * delta_t + jnp.sqrt(delta_t * sigma**2) * noise
    v_new = v + dv * delta_t

    return u_new, v_new

def run_simulation_with_splits(N, y0='wave', a=3, b=0.2, e=1e-2, Du=1, L=None,  sigma=0.05, stimulus_time=2000, delta_t=0.1, T=40000.0, output_times=20000, random_key=random.PRNGKey(seed), split_t=10):
    # Calculate the number of solver steps based on the total time and delta_t
    num_steps = int(T / delta_t)
    output_every = int(max(num_steps / output_times, 1))
    steps_per_split = int(stimulus_time/(delta_t))
    num_splits= int(T/(stimulus_time))


    
    u0 = jnp.zeros(N, dtype=jnp.float32)
    v0 = jnp.zeros(N, dtype=jnp.float32)
   
    # Initialize output arrays
    vs = jax.device_put(jnp.zeros((int(output_times/num_splits), N), dtype=jnp.float32))

    # Define the scan function
    
    def scan_fn(step, carry):
        u, v, key, vs = carry
        key, subkey = random.split(key)
        # Update variables
        u, v = FHN_step(u, v, N, a, b, e, Du, sigma, L, subkey, delta_t)
        #vs = vs.at[step % vs.shape[0]].set(u)
        # Store output if at the correct interval
        vs = lax.cond(
            step % output_every == 0,
            lambda vs: vs.at[step // output_every, :].set(u),
            lambda vs: vs,
            vs)
        del subkey
        jax.clear_caches()
        gc.collect()
       


        return (u, v, key, vs)

    # Run the simulation in splits
    output_file = f'V_values_m={m}_seed={seed}.pkl'
    #output_file = f'/home/ipellini/V_values_brain/V_values_m={m}_seed={seed}.pkl'
    key0 = random_key
    for split in range(num_splits):
        
        # Run the scan function for the current split
        u0, v0, key0, vs = lax.fori_loop(0, steps_per_split, scan_fn, (u0, v0, key0, vs))
        #log_memory_usage()
       
       
        if split >= split_t:
            print(vs.shape)
            with open(output_file, 'ab') as f:
                pickle.dump(np.array(vs), f)
        jax.clear_caches()
     
    return 0
    
    
    
    

L1=random_graph(N,k,J=m/k)

run_simulation_with_splits(N, L=L1, random_key=random.PRNGKey(seed), split_t=10)
