

import os, sys

rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NPROC"]="1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import numpy as np 
import gc
import pickle
from scipy.special import erfcinv
from scipy.sparse import diags, coo_matrix
from scipy import sparse
import jax.numpy as jnp
from jax import lax, vmap, jit
import jax.random as random
from jax.experimental import sparse
import argparse

import psutil


parser = argparse.ArgumentParser(description='Run simulation with specified seed and block.')
parser.add_argument('--seed', type=int, required=True, help='Random seed for simulation.')
parser.add_argument('--xi', type=float, required=True, help='Correlation length')
args = parser.parse_args()

seed = args.seed
xi = args.block
print(f"Seed: {seed}, xi: {xi}")



N_x = 201
N_y = 201
N = N_x * N_y

# exponential correlation function
def C_exponential(x,xi):
    """
        Exponential correlation function

        Parameters
        x : np.array
            Array of distance vectors (x1,x2,...,xn) of shape (N,d)
        xi : float
            Correlation length

    """
    # make sure this also works for d=1
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    return np.exp(-np.linalg.norm(x, axis=1)/xi)


 



def correlated_disorder(L, xi, seed=1001):
    """
    Generate two independent Gaussian random fields with exponential correlation restrain in the interv
    Parameters:
    L (int): The size of the grid (L+1 x L+1).
    xi (float): The correlation length.
    Returns:
    tuple: Two 2D jax.numpy arrays representing the independent Gaussian random fields.
    """
    x = jnp.arange(-L//2, L//2+1)
    y = jnp.arange(-L//2, L//2+1)
    X, Y = jnp.meshgrid(x, y)
    coords = jnp.vstack([X.ravel(), Y.ravel()]).T

    Cx = C_exponential(coords, xi).reshape(x.size, y.size)
    Ck_2d = np.fft.fft2(Cx)
    Sk_2d = jnp.abs(Ck_2d)

    key = random.PRNGKey(seed)
    key1, key2 = random.split(key)

    Gk_2d = random.normal(key1, Ck_2d.shape) * jnp.sqrt(Sk_2d) * (L+1)
    Gk_2d1 = random.normal(key2, Ck_2d.shape) * jnp.sqrt(Sk_2d) * (L+1)

    # inverse Fourier transform gives two real-valued Gaussian random fields (real and imaginary part)
    Gx_2d = np.fft.ifft2(Gk_2d + 1j * Gk_2d1)
    Gx1_2d = np.real(Gx_2d)
    Gx2_2d = np.imag(Gx_2d)

    return Gx1_2d, Gx2_2d

def compute_theta(p, sigma_tau=1):
    """
    Computes theta given probability p and standard deviation sigma_tau.
    
    Parameters:
        p (float): Probability value (0 < p < 1).
        sigma_tau (float): Standard deviation.
        
    Returns:
        float: Corresponding theta value.
    """
    if not (0 < p < 1):
        raise ValueError("p must be in the range (0,1)")
    
    return erfcinv(2 * p) * np.sqrt(2 * sigma_tau**2)


def generate_laplacian(N, conduction_block_threshold,xi, seed, sparse_matrix=False):
    """
    Generate the Laplacian matrix for a grid graph with conduction blocks given by correlated disorder.
    parameters:
    N : int
        The size of the grid (N x N).
    conduction_block_threshold : float in [0, 1]
        The threshold for conduction blocks.
    xi : float
        The correlation length of the disorder.
    sparse_matrix : bool
        Whether to return a sparse matrix.
    returns:
    scipy.sparse.coo_matrix or np.ndarray
        The Laplacian matrix.
    np.ndarray
        The conduction blocks.  
    """
    num_nodes = N * N
    adj_rows = []
    adj_cols = []
    adj_data = []
    G2, Gx=correlated_disorder(N-1,xi, seed=seed)

    if conduction_block_threshold==0:
        conduction_blocks = np.zeros((N, N), dtype=bool)
    else:
        conduction_blocks = Gx.reshape(N, N) > compute_theta(conduction_block_threshold)



    # Function to map grid (i, j) to a single node index
    def node_index(i, j):
        return i * N + j

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
    indices = np.array([[i, j] for i in range(N) for j in range(N)])
    idx = node_index(indices[:, 0], indices[:, 1])

    for di, dj, weight in neighbors:
        ni = indices[:, 0] + di
        nj = indices[:, 1] + dj

    # Step 1: Filter for in-bounds neighbors
        in_bounds = (ni >= 0) & (ni < N) & (nj >= 0) & (nj < N)
    
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


    if sparse_matrix:

        return sparse.BCSR.from_scipy_sparse(laplacian_matrix), jnp.array(conduction_blocks)
    
    else:
        return laplacian_matrix.todense(), conduction_blocks






def FHN_step(v, w, N, a, b, e, Dv, sigma, L, key, delta_t):

    # Generate Gaussian noise for each element of h
    noise = random.normal(key, v.shape)
    
    dv = a * v * (v - b) * (1 -v) - Dv * (L @ v) - w 
    dw = e * (v - w)
    v_new = v + dv * delta_t + jnp.sqrt(delta_t * sigma**2) * noise
    w_new = w + dw * delta_t

    return v_new, w_new

def run_simulation_with_splits(N,  a=3, b=0.05, e=1e-2, Dv=0.04, L=None, indices=None, sigma=0.0001, stimulus_time=2000, delta_t=0.1, T=8000.0, output_times=4000, random_key=random.PRNGKey(0), split_t=2):
    # Calculate the number of solver steps based on the total time and delta_t
    num_steps = int(T / delta_t)
    output_every = int(max(num_steps / output_times, 1))
    steps_per_split = int(stimulus_time / delta_t)
    num_splits= int(T/stimulus_time)
 

    v0 = jnp.zeros(N, dtype=jnp.float32)
    w0 = jnp.zeros(N, dtype=jnp.float32)
    v0 = v0.at[indices].set(0.1)

    # Initialize output arrays
    vs = jnp.empty((int(output_times / num_splits), N), dtype=jnp.float32)
    
    # Define the scan function
    def scan_fn(step, carry):
        v, w, key, vs = carry
        key, subkey = random.split(key)
        # Update variables
        v, w = FHN_step(v, w, N, a, b, e, Dv, sigma, L, subkey, delta_t)

        # Store output if at the correct interval
        vs = lax.cond(
            step % output_every == 0,
            lambda vs: vs.at[step // output_every, :].set(v),
            lambda vs: vs,
            vs
        )

        return (v, w, key, vs)

    # Run the simulation in splits
    output_file = f'/scratch01.local/ipellini/V_values_xi=20/V_values_xi={xi}_p={i}_seed={seed}.pkl'
    key0 = random_key
    for split in range(num_splits):

        # Run the scan function for the current split
        v0, w0, key0, vs = lax.fori_loop(0, steps_per_split, scan_fn, (v0, w0, key0, vs))
        #print(f"Split: {split}, Memory usage of vs: {vs.nbytes / 1024**2:.2f} MB")
        v0 = v0.at[indices].add(0.1)
    
        if split >= split_t:
        
            with open(output_file, 'ab') as f:
                pickle.dump(vs, f)

    return 0
    

for i in np.arange(0,1,0.01):
    i=round(i,2)   
    

    L1, c1 = generate_laplacian(N_x,N_y,  i, xi, seed,sparse_matrix=True,)
    indices = jnp.where((jnp.arange(N) % N_x == 0) & (c1.flatten() == 0))[0]

    run_simulation_with_splits(N, L=L1, indices=indices, random_key=random.PRNGKey(seed), split_t=2)
