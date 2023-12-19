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
import pickle
import random
import numpy as np

class FeatureFunction( object ):
    """
    FeatureFunction is the function phi(s,a) that captures all the linear
    features in state(s) by taking action(a).
    It maps each state-action pair to a vector of feature values, x_vector.
    For example, if there are two actions, a1 and a2:
    phi(s,a1) --> x_vector = [p1,p2,p3, 0,0,0,    1] # with bias term
    phi(s,a2) --> x_vector = [0,0,0,    p1,p2,p3, 1]
    """
    
    def __init__(self, sim, name='Proportional', init_w_val=None):
        """
        Default FeatureFunction is proportional in each of the continuous
        state parameters.
        
        paramL is a list of ContinuousParameter objects describing each state
        parameter.
        
        if init_w_val is None, initialize to random values
        """
        
        self.name = name
        self.sim = sim
        self.paramL = sim.paramL
        self.full_actionL = sim.get_full_action_list()
        self.Nactions = len(self.full_actionL)
        
        self.actionD = {} # index=a_desc, value=index into full_actionL
        for i,a in enumerate(self.full_actionL):
            self.actionD[a] = i
        
        self.init_w_val = init_w_val # if None then initialize to random values
        self.init_w_vector()
        
    
    # ======================== OVERRIDE STARTING HERE ==========================
    def desc(self):
        return self.name
    
    def init_w_vector(self):
        """
        Initialize the weights vector and the number of entries, N.
        NOTE: bias term is included in N.
        """
        
        num_w_per_action = len(self.paramL) + 1 # for simple Proportional model with bias term
        
        # initialize a weights numpy array with random values.
        N =  num_w_per_action * self.Nactions
        
        if self.init_w_val is None:
            self.w_vector = np.random.randn(N) / np.sqrt(N)
        else:
            self.w_vector = np.array( [self.init_w_val]*N )
            
        self.N = len( self.w_vector )
        self.num_w_per_action = num_w_per_action
    
    def get_x_terms_for_an_action(self, s_vector):
        """For Proportional, x_terms_for_an_action == s_vector."""
        L = [v for v in s_vector]
        L.append(1.0)
        return np.array( L )
    
    def get_x_vector(self, a_desc, s_vector=None ):
        """
        Return the x vector (feature vector) that represents the state, s_vector.
        
        NOTE: if s_vector is None, then assume self.paramL holds current state
        """
        if s_vector is None:
            s_vector = np.array( [p.value for p in self.paramL] )
        
        x_vector = np.zeros( self.N )
        #x_vector[-1] = 1.0 # set bias term
        #print('x_vector:', x_vector, type(x_vector), '  shape:',x_vector.shape)
        
        x_terms_for_an_action = self.get_x_terms_for_an_action( s_vector )
        #print('x_terms_for_an_action:',x_terms_for_an_action, 
        #      type(x_terms_for_an_action), '  shape:',x_terms_for_an_action.shape)
        i = self.actionD[a_desc]*self.num_w_per_action
        #print('self.actionD', self.actionD)
        #print('i:',i)
        
        # for Proportional assumption, each x value is equal to each s value
        # (note a copy of s_vector for each possible action... self.Nactions)
        x_vector[i:i+self.num_w_per_action] = x_terms_for_an_action
        x_vector[-1] = 1.0 # set bias term
        #print('From get_x_vector:', x_vector)
        return x_vector
    # ======================== OVERRIDE ENDING HERE ==========================

    def get_QsaEst(self, a_desc, s_vector=None):
        """Return the current estimate for Q(s,a) from linear function eval."""
        
        x_vector = self.get_x_vector( a_desc, s_vector=s_vector )
        #print('w_vector:',self.w_vector)
        #print('x_vector:',x_vector)
        return self.w_vector.dot( x_vector )

    #def get_QsaEst_from_x_vector(self, x_vector):
    #    """Return the current estimate for Q(s,a) from linear function eval."""
    #    return self.w_vector.dot( x_vector )
    
    def get_gradient(self, a_desc, s_vector=None):
        """
        Return the gradient of value function with respect to w_vector.
        Since the function is linear in w, the gradient is = x_vector.
        
        NOTE: if s_vector is None, then assume self.paramL holds current state
        """
        return self.get_x_vector( a_desc, s_vector )

    # ========================== pickle routines ===============================

    def make_pickle_filename(self, fname):
        """Make a file name ending with .w_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.w_pickle'

        else:
            fname = fname.replace(' ','_').replace('.','_') + '.w_pickle'

        return fname

    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        # build name for pickle
        fname = self.make_pickle_filename( fname )

        saveD = {}
        saveD['w_vector'] = self.w_vector

        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
        print('Saved ActionValueColl to file:',fname)

    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle file."""

        fname = self.make_pickle_filename( fname )
        if not os.path.isfile( fname ):
            print('Pickle File NOT found:', fname)
            return None

        try:
            fileObject = open(fname,'rb')
            readD = pickle.load(fileObject)

            w_vector = readD['w_vector']

            fileObject.close()
            print('Read ActionValueColl from file:',fname)

            return w_vector
        except:
            return None

    def init_from_pickle_file(self, fname=None): # pragma: no cover
        """Initialize ActionValueColl from policy pickle file."""
        w_vector = self.read_pickle_file( fname=fname )
        if w_vector is not None:
            self.w_vector = w_vector
        else:
            print('ERROR... Failed to read file:', fname)

        

if __name__=="__main__":
    
    from introrl.continuous_sims.sim_continuous import ContinuousSimulation
    
    sim = ContinuousSimulation(name='Mountain Car', step_reward=-1.0)
        
    ff = FeatureFunction( sim, name='Proportional', init_w_val=None)
    
    print(ff.get_gradient(1))
    print(ff.get_QsaEst(1))
        