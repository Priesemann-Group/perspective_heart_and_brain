import numpy as np 



import pickle
import os, sys
import gc
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)



from scipy.sparse import diags, coo_matrix
from scipy import sparse
import jax.numpy as jnp
from jax import lax, vmap, jit
import jax.random as random
from jax.experimental import sparse
import argparse

parser = argparse.ArgumentParser(description='Run simulation with specified seed and block.')
parser.add_argument('--seed', type=int, required=True, help='Random seed for simulation.')
parser.add_argument('--block', type=float, required=True, help='Conduction block threshold.')
args = parser.parse_args()

seed = args.seed
block = args.block
print(f"Seed: {seed}, Block: {block}")



N_x = 200
N_y = 200
N = N_x * N_y


def generate_laplacian(N, M, conduction_block_threshold, sparse_matrix=False, seed=0):
    num_nodes = N * M
    adj_rows = []
    adj_cols = []
    adj_data = []

    # Generate random conduction blocks
    np.random.seed(seed)
    conduction_blocks = np.random.rand(N, M) < conduction_block_threshold

    # Function to map grid (i, j) to a single node index
    def node_index(i, j):
        return i * M + j

    # Define neighbors for the nine-point stencil with weights
    neighbors = [
        (-1, 0, .5),     # up
        (1, 0, .5),      # down
        (0, -1, .5),     # left
        (0, 1, .5),      # right
        (-1, -1, .25),   # top-left
        (-1, 1, .25),    # top-right
        (1, -1, .25),    # bottom-left
        (1, 1, .25)      # bottom-right
    ]
    
    # Build adjacency structure excluding conduction blocks
    indices = np.array([[i, j] for i in range(N) for j in range(M)])
    idx = node_index(indices[:, 0], indices[:, 1])

    for di, dj, weight in neighbors:
        ni = indices[:, 0] + di
        nj = indices[:, 1] + dj

    # Step 1: Filter for in-bounds neighbors
        in_bounds = (ni >= 0) & (ni < N) & (nj >= 0) & (nj < M)
    
    # Step 2: Find valid indices (in-bounds) to avoid shape mismatches
        valid_indices = np.where(in_bounds)[0]
        ni_valid = ni[valid_indices]
        nj_valid = nj[valid_indices]

    # Step 3: Apply conduction block exclusion on the filtered indices
        valid_conduction = ~conduction_blocks[ni_valid, nj_valid]
        valid_node = ~conduction_blocks[indices[valid_indices, 0], indices[valid_indices, 1]]
        valid = valid_conduction & valid_node

    # Step 4: Append data for fully valid connections
        adj_rows.extend(idx[valid_indices][valid])
        adj_cols.extend(node_index(ni_valid[valid], nj_valid[valid]))
        adj_data.extend([weight] * int(np.sum(valid)))


    # Create adjacency and degree matrices
    adj_matrix = coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(num_nodes, num_nodes))
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_matrix = diags(degrees)

    # Construct Laplacian matrix
    laplacian_matrix = degree_matrix - adj_matrix

    del indices, adj_rows, adj_cols, adj_data, ni, nj, valid_indices, ni_valid, nj_valid, valid_conduction, valid_node, valid
    gc.collect()  # Force garbage collection

    if sparse_matrix:

        return sparse.BCSR.from_scipy_sparse(laplacian_matrix), jnp.array(conduction_blocks)
    
    else:
        return laplacian_matrix.todense(), conduction_blocks


def FHN_step(u, v, N, a, b, e, Du, sigma, L, key, delta_t):

    # Generate Gaussian noise for each element of h
    noise = random.normal(key, u.shape)
    
    du = a * u * (u - b) * (1 - u) - Du * (L @ u) - v 
    dv = e * (u - v)
    u_new = u + du * delta_t + jnp.sqrt(delta_t * sigma**2) * noise
    v_new = v + dv * delta_t

    return u_new, v_new

def run_simulation_with_splits(N,  a=3, b=0.05, e=1e-2, Du=0.04, L=None, indices=None, sigma=0.0001, stimulus_time=1000, delta_t=0.1, T=4000.0, output_times=1333, random_key=random.PRNGKey(0), split_t=2):
    # Calculate the number of solver steps based on the total time and delta_t
    num_steps = int(T / delta_t)
    output_every = int(max(num_steps / output_times, 1))
    steps_per_split = int(stimulus_time / delta_t)
    num_splits= int(T/stimulus_time)
 

    u0 = jnp.zeros(N, dtype=jnp.float32)
    v0 = jnp.zeros(N, dtype=jnp.float32)
    u0 = u0.at[indices].set(0.1)

    # Initialize output arrays
    vs = jnp.empty((int(output_times / num_splits), N), dtype=jnp.float32)

    # Define the scan function
    @jit
    def scan_fn(step, carry):
        u, v, key, vs = carry
        key, subkey = random.split(key)
        # Update variables
        u, v = FHN_step(u, v, N, a, b, e, Du, sigma, L, subkey, delta_t)

        # Store output if at the correct interval
        vs = lax.cond(
            step % output_every == 0,
            lambda vs: vs.at[step // output_every, :].set(u),
            lambda vs: vs,
            vs
        )

        return (u, v, key, vs)

    # Run the simulation in splits
    output_file = f'V_values_p={block}_seed={seed}.pkl'
    key0 = random_key
    for split in range(num_splits):

        # Run the scan function for the current split
        u0, v0, key0, vs = lax.fori_loop(0, steps_per_split, scan_fn, (u0, v0, key0, vs))

        u0 = u0.at[indices].add(0.1)
        
        if split >= split_t:
            print(vs.shape)
            with open(output_file, 'ab') as f:
                pickle.dump(vs, f)

    return 0
    

    
    

L1, c1 = generate_laplacian(N_x,N_y, block,sparse_matrix=True, seed=seed)
indices = jnp.where((jnp.arange(N) % N_x == 0) & (c1.flatten() == 0))[0]
run_simulation_with_splits(N, L=L1, indices=indices, random_key=random.PRNGKey(seed), split_t=2)
