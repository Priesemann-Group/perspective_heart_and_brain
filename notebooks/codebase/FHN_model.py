import networkx as nx
import numpy as np #we might be able to get rid of this if we implement better laplacian for heart
from scipy.sparse import diags, coo_matrix
from scipy import sparse
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import sparse
from jax import lax, vmap

import diffrax
import lineax

class FHN_model:
    def __init__(self, 
                N=200, 
                a=3, 
                b=None, 
                e=0.01, 
                noise_amp=None, 
                v0='zeros', 
                w0='zeros', 
                L=None, 
                m=0.005,
                random_key=jr.PRNGKey(1000), 
                v0_noise_amp=None, 
                w0_noise_amp=None,
                organ='brain',
                p=0,
                Du=None,
                k=50,
                Laplacian_seed=1000,
                stimulus_time=1000):
        # Model Parameters
        self.N = N
        self.a = a
        self.e = e
        self.L = L
        self.organ=organ
        self.p=p
        self.Laplacian_seed=Laplacian_seed
        self.stimulus_time=stimulus_time
        self.k=k
        self.m=m

        # TODO atm here I pass only the parameters we use in the simulations. For educational purposes we should change it
        if type(organ)==str and organ == 'brain':
            Du = 1
            noise_amp = 0.1
            b=0.2
            self.block= None
            self.initiate_random_graph()
            # Solution objects
            if type(v0) == str and v0=='zeros':
                v0 = jnp.zeros(N)
            if type(w0) == str and w0=='zeros':
                w0 = jnp.zeros(N)
        
            if type(v0) == str and v0=='random':
                v0 = jax.random.normal(random_key, shape=(2*N,))*v0_noise_amp
            if type(w0) == str and w0=='random':
                v0 = jax.random.normal(random_key, shape=(2*N,))*w0_noise_amp
        elif type(organ)==str and organ == 'heart':
            Du = -0.04
            noise_amp = 0.0001
            b=0.05
            self.generate_laplacian()
            indices = jnp.where((jnp.arange(N*N) % N == 0) & (self.block.flatten() == 0))[0]
            y0 = jnp.zeros(2 * N*N, dtype=jnp.float32)
            y0 = y0.at[indices].set(0.1)
            v0=y0[:N*N]
            w0=y0[N*N:]
        else:
            raise ValueError('Organ must be either brain or heart')


        self.Du = Du
        self.noise_amp = noise_amp
        self.b=b


        self.v0 = v0
        self.w0 = w0
        self.ts = None
        self.vs = None
        self.ws = None
        

    def nullclines(self, v_array):
        return (v_array, self.a*v_array*(v_array-self.b)*(1-v_array))  # return the corresponding w_arrays
    
    #random network implementation TODO: implement how to pass gaussian weights
    def initiate_random_graph(self,  seed=100, weights='homogeneous', weight_generator=None):
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
        p = self.k / (self.N - 1)
        J=self.m/self.k
        # Create ER graph
        G = nx.erdos_renyi_graph(self.N, p, directed=True, seed=seed)
        
        # Put weights
        for u, v in G.edges():
            if weights=='homogeneous':
                G[u][v]['weight'] = J#generator.normal(J, J/5)
            elif weights=='gaussian':
                G[u][v]['weight'] = weight_generator.normal(J[0], J[1])
        
        # Get the adjacency matrix in sparse format
        adj_matrix = nx.adjacency_matrix(G, weight='weight')
        
        self.L = sparse.BCSR.from_scipy_sparse(adj_matrix)
    
    #Laplacian generation for the heart
    def generate_laplacian(self, sparse_matrix=True):
        #TODO: implement fully in jax
        num_nodes = self.N * self.N
        adj_rows = []
        adj_cols = []
        adj_data = []

        # Generate random conduction blocks
        np.random.seed(self.Laplacian_seed)
        conduction_blocks = np.random.rand(self.N, self.N) < self.p

        # Function to map grid (i, j) to a single node index
        def node_index(i, j):
            return i * self.N + j

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
        indices = np.array([[i, j] for i in range(self.N) for j in range(self.N)])
        idx = node_index(indices[:, 0], indices[:, 1])

        for di, dj, weight in neighbors:
            ni = indices[:, 0] + di
            nj = indices[:, 1] + dj

        # Step 1: Filter for in-bounds neighbors
            in_bounds = (ni >= 0) & (ni < self.N) & (nj >= 0) & (nj < self.N)
    
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

            self.L= sparse.BCSR.from_scipy_sparse(laplacian_matrix)
            self.block= jnp.array(conduction_blocks)
    
        else:
            self.L= laplacian_matrix.todense()
            self.block= conduction_blocks

    # Deterministic part of the differential equation
    def FHN_graph(self, v, w):
        dv = self.a*v*(v-self.b)*(1-v) + self.Du*(self.L @ v) - w 
        dw = self.e*(v-w)

        return (dv,dw)
    
    # def FHN_graph(v, w, params):
    #     a, b, e, L, _, _ = params
    #     dv = a*v*(v-b)*(1-v) + L@v - w 
    #     dw = e*(v-w)

    #     return (dv,dw)

    # Stochastic part of the differential equation
    def FHN_graph_noise(self):
        noise = self.noise_amp*jnp.ones(self.N)
        return noise
    # def FHN_graph_noise(params):
    #     _, _, _, _, noise_amp, N = params
    #     noise = noise_amp*jnp.ones(N)
    #     return noise
      
    
    # def solve_with_diffrax(self, T=1000, max_steps=1000000, output_times=1000, solver=diffrax.ShARK(),rtol=1e-2, atol=1e-4,dt0=1e-2, noise_tol=1e-4, pcoeff=0.1, icoeff=0.0, dcoeff=0, random_key=jr.PRNGKey(0)):
    #     '''
    #     Takes in all simulation and solver parameters and returns a solution of the coupled SDEs

    #     -- CURRENTLY NOT WORKING ---

    #     Parameters:
    #         TODO
            
    #     Returns:
    #         diffrax.Solution object
    #     '''

    #     # Wrapper function that translate the diff eq functions to the format diffrax expects
    #     def FHN_graph_wrapper(t, y, params):
    #         _, _, _, _, _, N = params
    #         v = y[:N]
    #         w = y[N:]
    #         return FHN_model.FHN_graph(v, w, params)
        
    #     def FHN_graph_noise(t, y, params):
    #         return lineax.DiagonalLinearOperator(FHN_model.FHN_graph_noise(params))
        
    #     # Set up the stochastic diff eq
    #     deterministic_term = diffrax.ODETerm(FHN_graph_wrapper)
    #     brownian_path = diffrax.VirtualBrownianTree(0., T, tol=noise_tol, shape=(2*self.N,), key=random_key,levy_area=diffrax.SpaceTimeLevyArea)
    #     noise_term = diffrax.ControlTerm(FHN_graph_noise, brownian_path)
    #     terms = diffrax.MultiTerm(deterministic_term, noise_term)

    #     # Set up the solver
    #     saveat = diffrax.SaveAt(ts=jnp.linspace(0, T, output_times))
    #     stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol, pcoeff=pcoeff, dcoeff=dcoeff, icoeff=icoeff)

    #     # Solve the diff. eq. 
    #     sol = diffrax.diffeqsolve(terms, solver, t0=0, t1=T, dt0=dt0, y0=jnp.append(self.v0, self.w0), args=(self.a,self.b,self.e,self.L,self.noise_amp,self.N), saveat=saveat, max_steps=max_steps, progress_meter=diffrax.TqdmProgressMeter(), stepsize_controller=stepsize_controller)
        
    #     # Solve results
    #     self.ts = sol.ts
    #     self.vs = sol.ys[:,:self.N]
    #     self.ws = sol.ys[:,self.N:]

    #     # Return solution object in case anything else is needed
    #     return sol
    
    
    
    

    
    
    
    def solve_with_EulerMaruyama(self, delta_t=0.1, T=3000.0, output_times=3000, random_key=jr.PRNGKey(0)): 
    

        # Calculate the number of solver steps based on the total time and delta_t
        num_steps = int(T / delta_t)

        # Initialize state variables
        v = self.v0
        w = self.w0
        if self.organ=='brain':
            # Define the scan function
            def scan_fn(carry, step):
                v, w, key = carry
                key, subkey = jr.split(key)

                # Update variables
                deterministic_update = self.FHN_graph(v, w)
                noise_update = jr.normal(subkey, v.shape) * self.noise_amp
                v = v + deterministic_update[0]*delta_t +  jnp.sqrt(delta_t) * noise_update
                w = w + deterministic_update[1]*delta_t
                return (v, w, key), (v, w)

            # Run the scan function
            (v, w, _), (v_trajectory, w_trajectory) = jax.lax.scan(scan_fn, (v, w, random_key), None, length=num_steps)

        if self.organ=='heart':
                Ntot=self.N*self.N
                indices = jnp.where((jnp.arange(Ntot) % self.N== 0) & (self.block.flatten() == 0))[0]
                
                def scan_fn(carry, step):
                    v, w, key= carry
                    key, subkey = jr.split(key)
                    
                    
                    
                     # Apply stimulus to the specified indices
                    v = jax.lax.cond((step > 0) & (step % int(self.stimulus_time / delta_t) == 0),
                                      lambda v: v.at[indices].add(0.1),
                                      lambda v: v,
                                      v)
                    deterministic_update = self.FHN_graph(v, w)
                    noise_update = jr.normal(subkey, v.shape) * self.noise_amp
                    v = v + deterministic_update[0]*delta_t +  jnp.sqrt(delta_t) * noise_update
                    w = w + deterministic_update[1]*delta_t
    
    
                    return (v, w, key), (v, w)
                # Create a range of steps
                steps = jnp.arange(num_steps)
    
                # Run the scan function
                carry = (v, w, random_key)
                (v, w, _), (v_trajectory, w_trajectory) = jax.lax.scan(scan_fn, carry, steps)

        self.ts = jnp.linspace(0, T, output_times)
        output_every = int(max(num_steps/output_times,1))
        self.vs = v_trajectory[::output_every]
        self.ws = w_trajectory=w_trajectory[::output_every]
                
        return None
    
    def solve_with_EulerMaruyama_fori(self, delta_t=0.1, T=3000.0, output_times=3000, random_key=jr.PRNGKey(0)): 
              
        # Calculate the number of solver steps based on the total time and delta_t
        num_steps = int(T / delta_t)
        output_every = int(max(num_steps/output_times,1))
        if self.organ=='brain':
            # Initialize output arrays
            vs = jnp.zeros((output_times, self.N))
            ws = jnp.zeros((output_times, self.N))

            # Define the scan function
            def scan_fn(step, carry):
                v, w, key, vs, ws = carry
                key, subkey = jr.split(key)

                # Update variables
                deterministic_update = self.FHN_graph(v, w)
                noise_update = jr.normal(subkey, v.shape) * self.noise_amp
                v = v + deterministic_update[0]*delta_t +  jnp.sqrt(delta_t) * noise_update
                w = w + deterministic_update[1]*delta_t

                vs = vs.at[step//output_every,:].set(v)
                ws = ws.at[step//output_every,:].set(w)
                return (v, w, key, vs, ws)

            # Run the scan function
            _, _, _, vs, ws = jax.lax.fori_loop(0, num_steps, scan_fn, (self.v0, self.w0, random_key, vs, ws))
        if self.organ == 'heart':
            Ntot = self.N * self.N
            indices = jnp.where((jnp.arange(Ntot) % self.N == 0) & (self.block.flatten() == 0))[0]
            
            # Initialize output arrays
            vs = jnp.zeros((output_times, Ntot))
            ws = jnp.zeros((output_times, Ntot))
            # Define the scan function
            def scan_fn(step, carry):
                v, w, key, vs, ws = carry
                key, subkey = jr.split(key)
                # Apply stimulus to the specified indices
                v = jax.lax.cond((step > 0) & (step % int(self.stimulus_time / delta_t) == 0),
                        lambda v: v.at[indices].add(0.1),
                        lambda v: v,
                        v)
                deterministic_update = self.FHN_graph(v, w)
                noise_update = jr.normal(subkey, v.shape) * self.noise_amp
                v = v + deterministic_update[0] * delta_t + jnp.sqrt(delta_t) * noise_update
                w = w + deterministic_update[1] * delta_t
                vs = vs.at[step // output_every, :].set(v)
                ws = ws.at[step // output_every, :].set(w)
                return (v, w, key, vs, ws)
                # Run the scan function
            _, _, _, vs, ws = jax.lax.fori_loop(0, num_steps, scan_fn, (self.v0, self.w0, random_key, vs, ws))


        # Make sure only at most output_times many time points are stored
        self.ts = jnp.linspace(0, T, output_times)
        self.vs = vs
        self.ws = ws

        return None
    
    def EEG(self, random_sample=False):
        if random_sample:
            return self.vs[:, jr.randint(0,self.N,size=random_sample)].sum(axis=1).flatten()
        else:
            return self.vs.sum(axis=1).flatten()