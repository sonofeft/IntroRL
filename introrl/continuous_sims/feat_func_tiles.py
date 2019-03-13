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

class FeatFuncTiles( FeatureFunction ):
    """
    FeatureFunction is the function phi(s,a) that captures all the linear
    features in state(s) by taking action(a).
    It maps each state-action pair to a vector of feature values, x_vector.
    For example, if there are two actions, a1 and a2:
    phi(s,a1) --> x_vector = [p1,p2,p3, 0,0,0,    1] # with bias term
    phi(s,a2) --> x_vector = [0,0,0,    p1,p2,p3, 1]
    """
    
    def __init__(self, sim, name='TilingsInf', init_w_val=None, num_tiles=4, 
                 recenter=True, num_regionsL=None):
        
        self.num_tiles = num_tiles
        self.recenter = recenter
        
        if num_regionsL is None:
            num_regionsL = [8] * len(sim.paramL)
        
        self.num_regionsL = num_regionsL
        # create home tile 
        lo_valL = [p.min_value for p in sim.paramL]
        hi_valL = [p.max_value for p in sim.paramL]
        self.home_tile = Tile(lo_valL=lo_valL, hi_valL=hi_valL, num_regionsL=num_regionsL)
        
        self.tilings = Tilings( self.home_tile, num_tiles=num_tiles, recenter=recenter )
                     
        FeatureFunction.__init__(self,  sim, name=name, init_w_val=init_w_val)
    
    # ======================== OVERRIDE STARTING HERE ==========================
    def desc(self):
        
        return self.name + '_N%i_'%self.num_tiles + '_'.join( ['%i'%n for n in self.num_regionsL] )
        
    def init_w_vector(self):
        """
        Initialize the weights vector and the number of entries, N.
        NOTE: bias term is included in N.
        """        
        num_w_per_action = len( self.tilings.get_numpy_encoding( self.sim.get_s_vector() ) )
        
        # initialize a weights numpy array with random values.
        N = num_w_per_action*self.Nactions  + 1 #  + 1 for bias term
        
        if self.init_w_val is None:
            self.w_vector = np.random.randn(N) / np.sqrt(N)
        else:
            self.w_vector = np.array( [self.init_w_val]*N )
            
        self.N = len( self.w_vector )
        self.num_w_per_action = num_w_per_action
    
    def get_x_terms_for_an_action(self, s_vector):
        """return array of n_output_features."""
        x_vector = self.tilings.get_numpy_encoding( s_vector )
        return x_vector
    
    def get_x_vector(self, a_desc, s_vector=None ):
        """
        Return the x vector (feature vector) that represents the state, s_vector.
        
        NOTE: if s_vector is None, then assume self.paramL holds current state
        """
        if s_vector is None:
            s_vector = np.array( [p.value for p in self.paramL] )
        
        x_vector = np.zeros( self.N )
        
        x_terms_for_an_action = self.get_x_terms_for_an_action( s_vector )
        i = self.actionD[a_desc]*self.num_w_per_action
                
        # for Proportional assumption, each x value is equal to each s value
        # (note a copy of s_vector for each possible action... self.Nactions)
        x_vector[i:i+self.num_w_per_action] = x_terms_for_an_action
        x_vector[-1] = 1.0 # set bias term
        return x_vector
    # ======================== OVERRIDE ENDING HERE ==========================


if __name__=="__main__":
    
    from introrl.continuous_sims.sim_continuous import ContinuousSimulation
    
    sim = ContinuousSimulation(name='Mountain Car', step_reward=-1.0)
        
    ff = FeatFuncTiles( sim,  init_w_val=0.0)
    print('-'*66)
    print(ff.get_gradient(1))
    print(ff.get_QsaEst(1))
        