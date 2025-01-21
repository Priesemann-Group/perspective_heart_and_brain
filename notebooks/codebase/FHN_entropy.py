from .FHN_model import *


class FHN_entropy:
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


    def pattern_entropy(self, binary_arrays, s, size=None):
        """
        Calculate the entropy of the patterns contained in a set of binary arrays.
        Parameters:
        binary_arrays (jax.numpy.ndarray): A 2D array of binary arrays (N, T).
        s (int): The length of each binary array.
        Returns:
            tuple: A tuple containing:
                - float: The entropy of the binary arrays.
                - float: The normalized entropy of the binary arrays.
                - int: The length of each binary array.
        """
         # Calculate the number of unique patterns and their counts
        unique_patterns, counts = jnp.unique(binary_arrays, axis=1, return_counts=True, size=size)
        total_patterns = jnp.sum(counts)


        probabilities = (counts+1) / (total_patterns+2**s)
        unobserved= 2**s- len(counts)
    
        entropy_obs = -jnp.sum(probabilities * jnp.log2(probabilities))
        entropy_un=-unobserved/(total_patterns+2**s)*jnp.log2(1/(total_patterns+2**s))
        entropy=entropy_obs+entropy_un
    
    
        max_entropy = binary_arrays.shape[0]  # Maximum entropy for binary arrays of a given length
        normalized_entropy = entropy / max_entropy

        return entropy, normalized_entropy
            
    def handling_subsets(self, array, s, threshold=0.08, Tin=100): #TODO fix the problem with Tin, try to make unified code for both or
        """
        This function binarizes the input array based on a threshold value, reshapes it, 
        and then calculates the entropy and normalized entropy using the `calculate_entropy_jax` function.
        Parameters:
           array (ndarray): The input array to be binarized and analyzed.
           s (int): The size parameter used for reshaping the array (length of one side of the box).
        Returns:
           tuple: A tuple containing:
               - entropy (float): The calculated entropy of the binarized array.
               - normalised (float): The normalized entropy of the binarized array.
        """
        if self.organ=='heart':
            binary_v = jnp.where(array > threshold, 1, 0)
            binary_v=binary_v.reshape(s**2, -1)
            binary_v=binary_v[:, Tin:]
    
            entropy, normalised=self.pattern_entropy(binary_v,s*s, 1)
        if self.organ=='brain':
            binary_v = jnp.where(array > threshold, 1, 0)
            binary_v=binary_v.reshape(s, -1)
            binary_v=binary_v[:, Tin:]
    
            entropy, normalised=self.pattern_entropy(binary_v,s, 1)
        return entropy, normalised
    
    def entropy_calculation(self, frame_size=9):
        """
        Splits the input array into smaller sequences of size (frame_size, T)
        and calculates the entropy for each sequence using the entropycalc function.
    
        Parameters:
            array (jnp.ndarray): Input array of shape (N, T).
            frame_size (int): Size of the smaller sequences. Default is 9.
        
        Returns:
            jnp.ndarray: Array of entropies for each sequence.
            jnp.ndarray: Array of normalized entropies for each sequence.
        """
        if self.organ == 'brain': 
            array=self.vs.T  
            N, T = array.shape
            num_frames = N // frame_size

            def calculate_entropy_for_frame(i):
                frame = lax.dynamic_slice(array, (i, 0), (frame_size, T))
                return self.handling_subsets(frame, frame_size, threshold= 0.5)

            indices = jnp.arange(0, N, frame_size)
    
            entropies, normalized_entropies = vmap(calculate_entropy_for_frame)(indices)
            self.entropy=jnp.mean(normalized_entropies)
        if self.organ == 'heart':
            array=self.vs.T 
            frame_size=3
            # TODO : add option to remove boundary sites
            array=array.reshape(self.N, self.N, -1)
            N, _, T = array.shape
            num_frames = (N // frame_size) ** 2
            def calculate_entropy_for_frame(i, j):
    
                frame = lax.dynamic_slice(array, (i, j, 0), (frame_size, frame_size, T))
                return self.handling_subsets(frame, frame_size)
            indices = jnp.arange(0, N, frame_size)
    
            entropies, normalized_entropies = vmap( lambda i: vmap(lambda j: calculate_entropy_for_frame(i, j))(indices))(indices)
            self.entropy=jnp.mean(normalized_entropies)