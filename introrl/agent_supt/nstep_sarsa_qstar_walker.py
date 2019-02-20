#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from introrl.utils.circular_list import CircularList
from introrl.agent_supt.action_value_coll import ActionValueColl
from introrl.agent_supt.epsilon_calc import EpsilonGreedy

import sys

class NStepSarsaQStarFinder( object ):
    """
    Find the optimal policy by updating a ActionValueColl according to
    the n-step Sarsa algorithm from page 147 of Sutton&Barto 2nd Ed.
        
    When a terminal state is reached, or maximum number of steps is reached,
    do the final updates with ever-shortening from Nsteps, updates.
    
    Assume an eps_greedy for policy steps.
    Will call "eps_greedy.inc_N_episodes()" for any non-constant epsilon calcs.
    """
    
    def __init__(self, environment, Nsteps=3, 
                 epsilon=0.1, init_q_val=0.0,
                 terminal_set=None,
                 max_steps=sys.maxsize):
        
        self.environment = environment
        self.av_coll = ActionValueColl( environment, init_val=init_q_val )
        
        # assume a constant epsilon for now.
        self.epsgreedy_obj = EpsilonGreedy(epsilon=epsilon, 
                                           const_epsilon=True, half_life=200,
                                           N_episodes_wo_decay=0)
        
        if terminal_set is None:
            self.terminal_set = environment.terminal_set
        else:
            self.terminal_set = terminal_set
        
        self.Nsteps = Nsteps
        self.max_steps = max_steps
        
        self.clear() # initialize the (s,a,r) data structures, t=0, T=inf.
                        
    def clear(self):
        # The (s,a,r) data will be in circular lists such that the index will wrap-around.
        self.S = CircularList( [0] * (self.Nsteps+1) )
        self.A = CircularList( [0] * (self.Nsteps+1) )
        self.R = CircularList( [0] * (self.Nsteps+1) )
        
        self.t = 0 # current time value
        self.T = sys.maxsize # T initialized to infinity
        self.tau = 0 # will be position getting update
    
    def initialize(self, start_state_hash=None ):
        """
        initialize values at self.t
        (If start_state_hash is input use it, otherwise use environment.start_state_hash)
        """

        self.clear() # initialize the (s,a,r) data structures, t=0, T=inf.
        
        if start_state_hash is None:
            start_state_hash = self.environment.start_state_hash
        
        self.S[0] = start_state_hash

        a_desc = self.av_coll.get_best_eps_greedy_action( start_state_hash, 
                                                          epsgreedy_obj=self.epsgreedy_obj )
            
        if a_desc is None:
            self.T = 0 # ending before we start
            
        self.A[0] = a_desc
        
        sn_hash, reward = self.environment.get_action_snext_reward( self.S[0], self.A[0] )
        self.S[1] = sn_hash
        self.R[1] = reward
        
        if (sn_hash is None) or (sn_hash in self.terminal_set):
            self.T = 1 # ends pretty quickly
        else:
            # add next action, A[1]
            a_desc = self.av_coll.get_best_eps_greedy_action(  self.S[1], 
                                                  epsgreedy_obj=self.epsgreedy_obj )
            self.A[1] = a_desc
            if a_desc is None:
                self.T = 1 # ending quickly
            
        self.tau = self.t - self.Nsteps + 1
    
            
    def add_step(self):
        """
        Add a step from the ActionValueColl and add it to the lists.
        Assume that self.t has been properly set.
        """

        a_desc = self.A[self.t]
            
        if not a_desc is None:
            sn_hash, reward = self.environment.get_action_snext_reward( self.S[self.t], self.A[self.t] )
            self.S[self.t+1] = sn_hash
            self.R[self.t+1] = reward
            
            if (sn_hash is None) or (sn_hash in self.terminal_set):
                self.T = self.t + 1 # terminal
            else:
                # add next action
                a_desc = self.av_coll.get_best_eps_greedy_action(  self.S[self.t+1], 
                                                      epsgreedy_obj=self.epsgreedy_obj )
                self.A[self.t+1] = a_desc
                if a_desc is None:
                            self.T = self.t + 1 # terminal
                
            
    def do_sarsa_action_value_updates(self, alpha=0.1, gamma=1.0,
                                      start_state_hash=None): # only used for policy, not episode_obj
        """
        Given an initialized NStepSarsaQStarFinder,
        Iterate through the returns for the episode
        
        Update the ActionValueColl, av_coll as part of the episode iteration.
        
        NOTE: The ActionValueColl will be updated as part of this method.
        """
        
        self.initialize( start_state_hash=start_state_hash )
        # should have t=0, T=infinity, tau=negative
        
        total_num_steps = 0
        
        while self.tau < self.T - 1:
            total_num_steps += 1
            if total_num_steps >= self.max_steps:
                break
            
            self.t += 1
            if self.t < self.T:
                # Take an action according to policy (or episode_obj)
                self.add_step()
                    
            self.tau = self.t - self.Nsteps + 1
            if self.tau >= 0:
            # ------------------------------
                G = 0.0
                g_pow = 1.0 # gamma**n
                #print('       R=',self.R)
                for i in range(self.tau+1, min(self.tau+self.Nsteps, self.T)+1 ):
                    G += g_pow * self.R[i]
                    g_pow *= gamma
                    #print('             at i=%i, R[i]=%g'%(i, self.R[i]))
                
                if self.tau + self.Nsteps < self.T:
                    gpow = gamma**self.Nsteps
                    G += g_pow * self.av_coll.get_val( self.S[ self.tau+self.Nsteps ], self.A[ self.tau+self.Nsteps ] )
                    
                delta = alpha * ( G - self.av_coll.get_val( self.S[ self.tau ], self.A[ self.tau ] ) )                
                self.av_coll.delta_update( s_hash=self.S[ self.tau ], a_desc=self.A[ self.tau ], delta=delta )
    
if __name__ == "__main__":  # pragma: no cover
    
    from introrl.mdp_data.simple_grid_world import get_gridworld    
    
    gridworld = get_gridworld()
    print('Using an episode_obj')
    
    NSQF = NStepSarsaQStarFinder( gridworld, Nsteps=6 )
    for _ in range(100):
        NSQF.do_sarsa_action_value_updates( alpha=0.1, gamma=0.9, start_state_hash=None)
        NSQF.do_sarsa_action_value_updates( alpha=0.1, gamma=0.9, start_state_hash=(2,2))
    #print()
    #gridworld.summ_print()
    print()
    NSQF.av_coll.summ_print( fmt_Q='%g', none_str='*', show_states=True,
                             show_last_change=True, show_policy=False )
    
    
    
