from .FHN_model import *


class FHN_coherence:
    def __init__(self, model=None, vs=None, ws=None, ts=None):
        if model is not None:
            self.ts = model.ts
            self.vs = model.vs
            self.ws = model.ws
            self.N = model.N
            self.block = model.block
            self.organ = model.organ
        elif vs is not None and ws is not None:
            self.vs = vs
            self.ws = ws
            self.ts = ts
            self.N = vs.shape[1]
        else:
            raise ValueError("Either model or vs and ws must be provided")


    def calculate_coherence(self,  V, window_size, step_size=1):
        """
        Calculate the coherence order parameter R_V(t) using a sliding window approach.
    
        Parameters:
            V (jax.numpy.DeviceArray): A (N, T) array where N is the number of neurons, and T is the number of
            window_size (int): Size of the sliding window (number of time steps).
            step_size (int): Step size for sliding the window (default is 1).
        
        Returns:
            average coherence
        """
        N, T = V.shape  # Number of neurons and time points
    
    # Function to calculate coherence for a single window
        def coherence_for_window(start_idx):
            V_window = lax.dynamic_slice(V, (0, start_idx), (N, window_size))  # Extract window
        
            # Step 1: Population mean at each time step
            V_bar = jnp.mean(V_window, axis=0)  # Shape: (window_size,)
        
            # Step 2: Numerator (variance of population mean over neurons)
            V_bar_squared_mean = jnp.mean(V_bar**2)  # ⟨V̄(t)²⟩
            V_bar_mean_squared = jnp.mean(V_bar)**2  # ⟨V̄(t)⟩²
            numerator = jnp.sqrt(V_bar_squared_mean - V_bar_mean_squared)
        
            # Step 3: Denominator (variance of individual neurons)
            V_squared_mean = jnp.mean(V_window**2, axis=1)  # ⟨Vᵢ(t)²⟩ over window
            V_mean_squared = jnp.mean(V_window, axis=1)**2  # ⟨Vᵢ(t)⟩² over window
            denominator = jnp.sqrt(jnp.mean(V_squared_mean - V_mean_squared))
        
            # Step 4: Coherence order parameter for the window
            return numerator / denominator
    
        # Calculate coherence for all windows using sliding indices
        window_starts = jnp.arange(0, T - window_size + 1, step_size)
        R_V_t = vmap(coherence_for_window)(window_starts)
    
        return jnp.mean(R_V_t)
    # calculates coherence for both organs
    def coherence(self, window_size=1000, step_size=1, Tin=100): #TODO : fix the problem with Tin

        v_values=self.vs.T
        if self.organ == 'brain':
            self.R_V=self.calculate_coherence(v_values[:, Tin:], window_size, step_size)
        if self.organ == 'heart':
            v_values=v_values.reshape(self.N,self.N, -1)
            #TODO : add option to remove boundary sites
            block=self.block.reshape(self.N,self.N)
            block=~block
            #TODO: implement this in JAX. I tried several times and failed
            R=[]
            for j in range((self.N)):  # Iterate over the correct dimension
                filtered_column = v_values[:, j, Tin:][block[:,j]]
                R.append(self.calculate_coherence(filtered_column, 500))

            R=jnp.array(R)
            self.R_V=jnp.mean(R)
            

