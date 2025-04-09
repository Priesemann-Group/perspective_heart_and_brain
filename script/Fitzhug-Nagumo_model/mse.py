
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
from scipy.special import erfcinv


import numpy as np 
import jax.random as random
import jax

import os, sys
rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)

from jax import lax
import pickle
import jax.numpy as jnp

import argparse

parser = argparse.ArgumentParser(description='Run simulation with specified seed and block.')
parser.add_argument('--seed', type=int, required=True, help='Random seed for simulation.')
args = parser.parse_args()

seed = args.seed

print(f"Seed: {seed}")


N_x=200
N_y=200
xi=0

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


def generate_laplacian_corr(N, conduction_block_threshold,xi, seed, sparse_matrix=False):
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
    G2, Gx=correlated_disorder(N-1,xi, seed=seed)

    conduction_blocks = Gx.reshape(N, N) > compute_theta(conduction_block_threshold)

    return jnp.array(conduction_blocks)
    


def generate_laplacian(N, M, conduction_block_threshold, seed=0):

    # Generate random conduction blocks
    np.random.seed(seed)
    conduction_blocks = np.random.rand(N, M) < conduction_block_threshold

    return jnp.array(conduction_blocks)
def compute_mse_across_chunks(data, t, c1):
    """
    Computes the Mean Squared Error (MSE) across trajectories of all chunks using JAX.
    
    Parameters:
    - data: jax.numpy array of shape (N, T) (N elements over T time steps)
    - t: chunk size (must evenly divide T)
    
    Returns:
    - MSE per node averaged across all chunks.
    """

    N, T = data.shape
    assert T % t == 0, "T must be evenly divisible by t"
    
    num_chunks = T // t
    reshaped = data.reshape(N, num_chunks, t)  # Shape: (N, num_chunks, t)
    
    def mse_pairwise(i, j):
        return jnp.mean((reshaped[:, i, :] - reshaped[:, j, :]) ** 2, axis=1)  # Shape: (N,)
    
    indices = [(i, j) for i in range(num_chunks) for j in range(i+1, num_chunks)]
    mse_values = jnp.array([mse_pairwise(i, j) for i, j in indices])  # Shape: (num_pairs, N)
    
    mse_per_node = np.mean(mse_values[:, ~c1.flatten()])  # Averaging over all pairs for False elements in c1
    return mse_per_node  # Returns MSE for each node (N)

for i in np.arange(0,1,0.01):
    i=round(i,2)
    c1=generate_laplacian(N_x, N_y, i, seed)
    output_file = f'V_values_p={i}_seed={seed}.pkl'
    simulation_data = read_simulation_data(output_file)
    simulation_data = simulation_data.T
    if simulation_data is not None:
        print(i, simulation_data.shape)
    output_file = f'/scratch01.local/ipellini/mse_unorrelates/mse_heart_seed={seed}.pkl'
    R=compute_mse_across_chunks(simulation_data, 2000,c1)
    R=float(R)
    # Prepare the data to be dumped
    data_to_dump = {
        'seed': seed,
        'p': i,
        'mse': R
    }

    # Write the header if the file does not exist
    if not os.path.exists(output_file):
        with open(output_file, 'wb') as f:
            pickle.dump(['seed', 'p', 'mse'], f)

    # Append the data to the file
    with open(output_file, 'ab') as f:
        pickle.dump(data_to_dump, f)
