#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import random
import numpy as np
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

from introrl.continuous_sims.feature_func import FeatureFunction
from introrl.utils.tiles_infinite import Tile, Tilings

class FeatFuncEngineeredMC( FeatureFunction ):
    """
    FeatureFunction is the function phi(s,a) that captures all the linear
    features in state(s) by taking action(a).
    It maps each state-action pair to a vector of feature values, x_vector.
    For example, if there are two actions, a1 and a2:
    phi(s,a1) --> x_vector = [p1,p2,p3, 0,0,0,    1] # with bias term
    phi(s,a2) --> x_vector = [0,0,0,    p1,p2,p3, 1]
    """
    
    def __init__(self, sim, name='Engineered_Mountain_Car', init_w_val=None):
                     
        FeatureFunction.__init__(self,  sim, name=name, init_w_val=init_w_val)
    
    # ======================== OVERRIDE STARTING HERE ==========================
    def desc(self):
        
        return self.name 
        
    def init_w_vector(self):
        """
        Initialize the weights vector and the number of entries, N.
        NOTE: bias term is included in N.
        """
        
        num_w_per_action = 1
        
        # initialize a weights numpy array with random values.
        N = self.Nactions * num_w_per_action  + 1 #  + 1 for bias term
        
        if self.init_w_val is None:
            self.w_vector = np.random.randn(N) / np.sqrt(N)
        else:
            self.w_vector = np.array( [self.init_w_val]*N )
            
        self.N = len( self.w_vector )
        self.num_w_per_action = num_w_per_action
    
    def get_x_terms_for_an_action(self, s_vector):
        """return array of n_output_features."""
        return s_vector
            
    def get_x_vector(self, a_desc, s_vector=None ):
        """
        Return the x vector (feature vector) that represents the state, s_vector.
        
        NOTE: if s_vector is None, then assume self.paramL holds current state
        """
        if s_vector is None:
            s_vector = np.array( [p.value for p in self.paramL] )
        
        x_vector = np.zeros( self.N )
        i = self.actionD[a_desc]*self.num_w_per_action
        
        x,v = s_vector
        if a_desc == 0:
            x_vector[i] = 1.0
        else:
            x_vector[i] = v
                
        x_vector[-1] = 1.0 # set bias term
        return x_vector
    # ======================== OVERRIDE ENDING HERE ==========================
    '''
        x,v = s_vector
        if v < -0.001:
            x_vector[0] = 1.0
        elif v > 0.001:
            x_vector[2] = 1.0
        else:
            x_vector[1] = 1.0
    '''
    def get_QsaEst(self, a_desc, s_vector=None):
        """Return the current estimate for Q(s,a) from linear function eval."""
        
        if s_vector is None:
            s_vector = np.array( [p.value for p in self.paramL] )
        
        x,v = s_vector
        v_limit = 0.0
        if a_desc==1:
            if v>v_limit:
                return 1.0
            else:
                return 0.0
        elif a_desc==-1:
            if v<0.0:
                return 1.0
            else:
                return 0.0
        elif a_desc==0:
            return 0.1
        

if __name__=="__main__":
    
    from introrl.continuous_sims.sim_continuous import ContinuousSimulation
    
    sim = ContinuousSimulation(name='Mountain Car', step_reward=-1.0)
        
    ff = FeatFuncEngineeredMC( sim,  init_w_val=0.0)
    print('-'*66)
    print(ff.get_gradient(1))
    print(ff.get_QsaEst(1))
        