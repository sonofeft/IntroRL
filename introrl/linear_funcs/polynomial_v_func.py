#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import os
import numpy as np
from introrl.linear_funcs.continuous_v_func import Continuous_V_Func

class Polynomial_V_Func( Continuous_V_Func ):
    """
    Create a function for an environment that encodes all of the states
    as a polynomial that is linear in the weights of the polynomial.
    
    e.g. degree=2 for 3 state vars --> [1, s1, s2, s3, s1^2, s1 s2, s1 s3, s2^2, s2 s3, s3^2]
    
    """
    
    # ======================== OVERRIDE STARTING HERE ==========================
    def init_w_vector(self):
        """Initialize the weights vector and the number of entries, N."""
        
        # initialize a weights numpy array with random values.
        N = len(self.sD)
        self.w_vector = np.random.randn(N) / np.sqrt(N)
        self.N = len( self.w_vector )
                
    def get_x_vector(self, s_hash ):
        """
        Return the x vector that represents the state, s_hash.
        NOTE: the index into x_vector for s_hash = self.sD[ s_hash ]
        """
        x_vector = np.zeros(self.N, dtype=np.float)
        x_vector[ self.sD[ s_hash ] ] = 1.0
        return x_vector
    # ======================== OVERRIDE ENDING HERE ==========================

    def VsEst(self, s_hash):
        """Return the current estimate for V(s) from linear function eval."""
        x_vector = self.get_x_vector( s_hash )
        return self.w_vector.dot( x_vector )
    
    def __init__(self, environment, polynomial_order=3):
        
        self.polynomial_order = polynomial_order
        
        Continuous_V_Func.__init__(self, environment)
        

if __name__ == "__main__": # pragma: no cover
    import sys
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()

    oh = Polynomial_V_Func( gridworld )
    


