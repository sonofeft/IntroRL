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

class FFPolynomial( FeatureFunction ):
    """
    FeatureFunction is the function phi(s,a) that captures all the linear
    features in state(s) by taking action(a).
    It maps each state-action pair to a vector of feature values, x_vector.
    For example, if there are two actions, a1 and a2:
    phi(s,a1) --> x_vector = [p1,p2,p3, 0,0,0,    1] # with bias term
    phi(s,a2) --> x_vector = [0,0,0,    p1,p2,p3, 1]
    """
    
    def __init__(self, sim, name='Polynomial', init_w_val=None, 
                 n_degree=2, interaction_only=False):
                     
        self.n_degree = n_degree
        self.interaction_only = interaction_only
        
        FeatureFunction.__init__(self,  sim, name=name, init_w_val=init_w_val)

    
    # ======================== OVERRIDE STARTING HERE ==========================
    def desc(self):
        if self.interaction_only:
            s = 'T'
        else:
            s = 'F'
        
        return self.name + '_degree%i'%self.n_degree + s
        
    def init_w_vector(self):
        """
        Initialize the weights vector and the number of entries, N.
        NOTE: bias term is included in N.
        """
        comb = (combinations if self.interaction_only else combinations_w_r)
        
        k_state_nums = len( self.sim.paramL )
        k = k_state_nums - 1 # NOTE: k_state_nums is 1-based indexing
        
        def get_combination_iter():
            return chain.from_iterable(comb(range(k_state_nums), i) for i in range( self.n_degree + 1))
            
        combinations = get_combination_iter()        
        self.powers = np.vstack([np.bincount(c, minlength=k_state_nums) for c in combinations])
        #self.powers = np.array([np.bincount(c, minlength=k_state_nums) for c in combinations])
        #print( 'powers =', repr(self.powers) )

        input_features = ['s%d' % i for i in range(1,self.powers.shape[1]+1)]
        #print( 'input_features =',input_features )
        # ----------------
        feature_nameL = []
        for row in self.powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join("%s^%d" % (input_features[ind], exp)
                                if exp != 1 else input_features[ind]
                                for ind, exp in zip(inds, row[inds]))
            else:
                name = "1"
            feature_nameL.append(name)
        print( feature_nameL )

        combinations = get_combination_iter()
        num_w_per_action = sum(1 for _ in combinations)
        #print('num_w_per_action =', num_w_per_action, '    (n+1)^k =',(self.n_degree+1)**k)
        
        
        # --------------------------------------
        
        # initialize a weights numpy array with random values.
        N = num_w_per_action*self.Nactions  + 1 #  + 1 for bias term
        
        if self.init_w_val is None:
            self.w_vector = np.random.randn(N) / np.sqrt(N)
        else:
            self.w_vector = np.array( [self.init_w_val]*N )
            
        self.N = len( self.w_vector )
        self.num_w_per_action = num_w_per_action
        
        #print('len(w_vector) =',len(self.w_vector) )
        #print('w_vector init =',self.w_vector)
        #print('   N=%i'%N, '   num_w_per_action=',self.num_w_per_action)
    
    def get_x_terms_for_an_action(self, s_vector):
        """return array of n_output_features."""
        x_vector = np.zeros( self.num_w_per_action )
        for i,pow in enumerate(self.powers):
            #print('self.powers=',self.powers,'   pow=',pow,'   s_vector=',s_vector)
            x_vector[i] = np.prod( np.power( s_vector, pow ) )
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
        
    ff = FFPolynomial( sim,  init_w_val=0.0)
    print('-'*66)
    print(ff.get_gradient(1))
    print(ff.get_QsaEst(1))
        