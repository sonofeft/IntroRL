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
    
    def F1(self, x,v):
        """Return func value for given x,v"""
        
        return v
        #v_limit = 0.0
        #if v>v_limit:
        #    return v
        #else:
        #    return 0.0
            
    def F2(self, x,v):
        """Return func value for given x,v"""
        
        v_limit = 0.01
        if abs(v)<v_limit:
            return v_limit - abs(v)
        else:
            return 0.0
    
    def init_w_vector(self):
        """
        Initialize the weights vector and the number of entries, N.
        NOTE: bias term is included in N.
        """
        
        num_w_per_action = 2
        
        # initialize a weights numpy array with random values.
        N = self.Nactions * num_w_per_action  + 1 #  + 1 for bias term
        
        if self.init_w_val is None:
            self.w_vector = np.random.randn(N) / np.sqrt(N)
        else:
            self.w_vector = np.array( [self.init_w_val]*N )
            
        self.w_vector = np.array( [-1.,1., 0.,0., 1.,1., 0.] )
        #self.w_vector = np.array( [-1., 0., 1., 0.] )
            
        self.N = len( self.w_vector )
        self.num_w_per_action = num_w_per_action
        #print('Initial w_vector:', self.w_vector)
    
    def get_x_terms_for_an_action(self, s_vector):
        """return array of n_output_features."""
        x,v = s_vector
        return np.array( [self.F1(x,v),] )
        #return np.array( [self.F1(x,v), self.F2(x,v)] )
            
    # ======================== OVERRIDE ENDING HERE ==========================
    '''
        
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
    '''

if __name__=="__main__":
    
    from introrl.continuous_sims.sim_continuous import ContinuousSimulation
    
    sim = ContinuousSimulation(name='Mountain Car', step_reward=-1.0)
        
    ff = FeatFuncEngineeredMC( sim,  init_w_val=0.0)
    print('-'*66)
    print('N:', ff.N)
    print(ff.get_gradient(1, [.1, .01]))
    print(ff.get_QsaEst(1, [.1, .01]))
        