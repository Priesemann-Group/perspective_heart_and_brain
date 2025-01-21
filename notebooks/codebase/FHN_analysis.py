

from .FHN_model import *
import numpy as np
from scipy.signal import hilbert

class FHN_analyzer:
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

    def EEG(self, random_sample=False, given_sample=None):
        if given_sample:
            return self.vs[:,given_sample].sum(axis=1)
        elif random_sample:
            return self.vs[:,np.random.randint(0,self.N,size=random_sample)].sum(axis=1)
        else:
            return self.vs.sum(axis=1)

    



















































    
