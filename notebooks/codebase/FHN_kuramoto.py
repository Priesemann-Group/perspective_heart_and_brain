from .FHN_model import *

from scipy.signal import hilbert

class FHN_kuramoto:
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

    
    def compute_phases(self, data):
        """
        Compute the phase of each element using the Hilbert transform.
    
        Parameters:
            data (jnp.ndarray): An (N, T) array where N is the number of elements and T is the time dimension.
        
        Returns:
            jnp.ndarray: An (N, T) array containing the phases.
        """
    
        analytic_signal = hilbert(data, axis=1)
   
        phases = jnp.angle(analytic_signal)
        return phases
    
    def kuramoto_order_parameter(self, phases):
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
    
        amplitude = jnp.abs(order_parameter_complex)
   
        phase = jnp.angle(order_parameter_complex)
        return amplitude, phase
    
    def kuramoto(self, Tin=1000, Tfin=2000):
        """
        Function that calculates Kuramoto distinguishing between the two organs TODO : find a smart way to exclude transient/last timesteps and cleaner heart code 
        """
        if self.organ == 'brain':
            phases=self.compute_phases(self.vs.T)  #check that I actually need the transpose
            amplitude, phase = self.kuramoto_order_parameter(phases[:,Tin:Tfin]) #here I somehow need to remove the transient and last timesteps
            self.R= jnp.mean(amplitude)
        if self.organ == 'heart':
            v_values= self.vs.T
            
            block= self.block
            block=block.reshape(self.N, self.N)
            if self.N>8:
                block=~block[4:(self.N-4), 4:(self.N-4)]  #remove borders to avoid boundary effects
                v_values=v_values.reshape(self.N, self.N, -1)
                v_values=v_values[4:(self.N-4), 4:(self.N-4), :]
                v_values=v_values.reshape(-1, v_values.shape[2])
                phases=self.compute_phases(v_values)
                phases=phases[:,Tin:Tfin]  #here to be changes to cover from equilibration to a bunch of stuff before end
                phases=phases.reshape(self.N-8,self.N-8,-1)
                # TODO: replace this loop
                R=[]
                for j in range(self.N-8):  # Iterate over the correct dimension

                    filtered_column = phases[:, j, :][block[:, j]]
    
                    r1, psi1=self.kuramoto_order_parameter(filtered_column)
                    R.append(r1)
                
                R=jnp.array(R)
 
                self.R= jnp.mean(R)
            else:
                block=~block
                phases=self.compute_phases(v_values)
                phases=phases[:,Tin:Tfin]  #here to be changes to cover from equilibration to a bunch of stuff before end
                phases=phases.reshape(self.N,self.N,-1)
                R=[]
                for j in range(self.N-8):  # Iterate over the correct dimension

                    filtered_column = phases[:, j, :][block[:, j]]
    
                    r1, psi1=self.kuramoto_order_parameter(filtered_column)
                    R.append(r1)
                
                R=jnp.array(R)
                self.R= jnp.mean(R)