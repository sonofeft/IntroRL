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
from introrl.continuous_sims.param_continuous import ContinuousParameter

class ContinuousSimulation( object ):
    """
    A Simulation of continuous floating point variables
    This Baseline Simulation is for a classic mountain car.
    
    see: for deep learning policy: https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2
    """

    def __init__(self, name='Mountain Car', step_reward=-1.0):
        
        self.name = name
        self.step_reward = step_reward
        
        self.init_param_list()
        self.paramD = {} # index=param name, value=ContinuousParameter object 
        for p in self.paramL:
            self.paramD[p.name] = p
            
        self.num_params = len( self.paramL )
    
    # =============== OVERRIDE STARTING HERE =========================
    def init_param_list(self):
        
        self.pos_cp = ContinuousParameter(name='Position', units='ft', value_init=-0.5,
                                          min_value=-1.2, max_value=0.6)
        self.vel_cp = ContinuousParameter(name='Velocity', units='ft/sec', value_init=0.0,
                                          min_value=-0.07, max_value=0.07)
                                     
        self.paramL = [self.pos_cp, self.vel_cp] # list of state ContinuousParameter objects
        #self.paramL = [self.vel_cp, self.pos_cp] # list of state ContinuousParameter objects

    def calc_dependent_states(self, s_vector):
        """Some models may have dependent state values"""
        return s_vector

    def get_action_snext_reward(self, a_desc, s_vector=None):
        """
        Return next state, s_vector, and reward
        """
        if s_vector is None:
            x,xdot = self.get_s_tuple()
        else:
            x,xdot = s_vector
            self.pos_cp.set_bounded_val( x )
            self.vel_cp.set_bounded_val( xdot )
        
        self.vel_cp.add_bounded_delta( 0.001*a_desc - 0.0025*np.cos(3*x) )
        
        self.pos_cp.add_bounded_delta( self.vel_cp.value )
        if self.pos_cp.at_min_limit():
            self.vel_cp.set_bounded_val( 0.0 )

        #if self.pos_cp.at_max_limit():
        #    reward = 10.0
        #else:
        reward = self.step_reward


        return self.get_s_vector(), reward
    
    def get_y_pos(self, x=None):
        if x is None:
            return np.sin( 3*self.pos_cp.value )
        else:
            return np.sin( 3*x )
    
    def is_terminal_state(self, s_vector=None):
        """
        Return True if current state is terminal.
        if s_vector is given, use it instead of current state.
        """        
        if s_vector is None:
            return self.pos_cp.at_max_limit()
        else:
            return s_vector[0] >= self.pos_cp.max_value
    
    def get_state_legal_action_list(self, s_vector=None):
        """
        Return a list of possible actions from current state.
        OR Empty list, if no actions.
        
        if s_vector is given, use it instead of current state.
        """        
        if self.is_terminal_state( s_vector ):
            return []
        else:
            return [-1,0,1]
            
    def get_full_action_list(self):
        return [-1,0,1]
    # =============== OVERRIDE ENDING HERE =========================
    
    def set_random_param_value(self, param_name='Position', lo_lim=-0.6, hi_lim=-0.4):
        self.paramD[param_name].set_bounded_val( lo_lim + random.random()*(hi_lim - lo_lim) )
    
    def set_current_state(self, s_vector):
        for val,p in zip(s_vector, self.paramL):
            p.set_bounded_val( val )
    
    def get_s_vector(self):
        return np.array( [p.value for p in self.paramL] )
    
    def get_s_tuple(self):
        return tuple( [p.value for p in self.paramL] )
        
    def reset(self):
        for p in self.paramL:
            p.reset()
    
    def summ_print(self):
                
        print('========= ContinuousSimulation: "%s" ========='%self.name)
        print('                   step_reward:', self.step_reward)
        
        for p in self.paramL:
            p.summ_print( pad='    ' )
        

if __name__=="__main__":
    
    MCar = ContinuousSimulation( name='Mountain Car', step_reward=-1.0)
    
    MCar.summ_print()
    print('%8.4f x %8.4f xdot'%MCar.get_s_tuple() )
    for _ in range(20):
        MCar.get_action_snext_reward( -1 )
        print('%8.4f x %8.4f xdot'%MCar.get_s_tuple(), '   actionL=',MCar.get_state_legal_action_list() )
    MCar.summ_print()
    
    print('random Position','v'*66)
    MCar.set_random_param_value( param_name='Position', lo_lim=-0.6, hi_lim=-0.4)
    MCar.summ_print()
    
    print('reset','v'*66)
    MCar.reset()
    s_vector = np.array( [0.0111111, 0.02222222] )
    MCar.set_current_state( s_vector )
    MCar.summ_print()
    
    
    