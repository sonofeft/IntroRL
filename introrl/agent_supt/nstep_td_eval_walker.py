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

import sys

class NStepTDWalker( object ):
    """
    Following the policy OR episode_obj, update a StateValueColl according to
    the n-step TD algorithm from page 144 of Sutton&Barto 2nd Ed.
        
    The calling routine is expected to maintain the policy.
    Any changes to the policy will be reflected in the next policy step(s) taken.
    
    When a terminal state is reached, or maximum number of steps is reached.
    Do the final updates with ever-shortening from Nsteps, updates.
    
    If eps_greedy is provided for policy steps, the calling routine is expected
    to call "eps_greedy.inc_N_episodes()" for any non-constant epsilon calcs.
    """
    
    def __init__(self, environment, Nsteps=3, 
                 policy=None, episode_obj=None, 
                 terminal_set=None,
                 max_steps=sys.maxsize, eps_greedy=None):
        
        self.environment = environment
        
        if terminal_set is None:
            self.terminal_set = environment.terminal_set
        else:
            self.terminal_set = terminal_set
        
        # if policy is input, then use it. Otherwise assume an episode_obj is input.
        self.policy = policy
        self.episode_obj = episode_obj
        if policy is None:
            self.use_policy = False
        else:
            self.use_policy = True
        
        self.Nsteps = Nsteps
        self.max_steps = max_steps
        self.eps_greedy = eps_greedy
        
        self.clear() # initialize the (s,a,r) data structures, t=0, T=inf.
            
    def set_episode_obj(self, episode_obj):
        """To start working on a new episode, call set_episode_obj."""
        self.episode_obj = episode_obj
        self.use_policy = False
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
        (If using policy, then start_state_hash is used, otherwise ignored.)
        """

        self.clear() # initialize the (s,a,r) data structures, t=0, T=inf.
        
        if self.use_policy:
            # use policy to do init
            self.S[0] = start_state_hash

            a_desc = self.policy.get_single_action( self.S[0] )
                
            if a_desc is None:
                self.T = 0 # ending before we start
                
            if self.eps_greedy is not None:
                #assumes incl_zero_prob=True
                legal_actionL = self.environment.get_state_legal_action_list( self.S[0] )
                a_desc = self.eps_greedy( a_desc, legal_actionL )
            self.A[0] = a_desc
            
            sn_hash, reward = self.environment.get_action_snext_reward( self.S[0], self.A[0] )
            self.S[1] = sn_hash
            self.R[1] = reward
            
            if (sn_hash is None) or (sn_hash in self.terminal_set):
                self.T = 1 # ends pretty quickly
        else:
            # use episode_obj to do init
            if len( self.episode_obj ) > 0:
                s_hash, a_desc, reward, sn_hash = self.episode_obj.get_step( 0 )
                self.S[0] = s_hash
                self.A[0] = a_desc
            
                self.S[1] = sn_hash
                self.R[1] = reward
            else:
                self.T = 0
                
        self.tau = self.t - self.Nsteps + 1
    
            
    def add_step(self):
        """
        Add a step from the policy OR episode_obj and add it to the lists.
        Assume that self.t has been properly set.
        """

        # Take an action according to policy (or episode_obj)
        if self.use_policy:
            a_desc = self.policy.get_single_action( self.S[self.t] )
                
            if a_desc is None:
                self.T = self.t # terminal
            else:
                if self.eps_greedy is not None:
                    #assumes incl_zero_prob=True
                    legal_actionL = self.environment.get_state_legal_action_list( self.S[self.t] )
                    a_desc = self.eps_greedy( a_desc, legal_actionL )
                self.A[self.t] = a_desc
                
                sn_hash, reward = self.environment.get_action_snext_reward( self.S[self.t], self.A[self.t] )
                self.S[self.t+1] = sn_hash
                self.R[self.t+1] = reward
                #print('    policy: reward=%g'%self.R[self.t+1], '  for state=%s'%str(self.S[self.t+1]), 
                #      '  term set=',self.terminal_set)
                
                if (sn_hash is None) or (sn_hash in self.terminal_set):
                    self.T = self.t + 1 # terminal
                    #print('    IN TERM SET:', sn_hash,'  at self.t=',self.t,'  ')
        else:
            # use episode_obj to do init
            if self.t < len( self.episode_obj ):
                s_hash, a_desc, reward, sn_hash = self.episode_obj.get_step( self.t )
                self.S[self.t] = s_hash
                self.A[self.t] = a_desc
            
                self.S[self.t+1] = sn_hash
                self.R[self.t+1] = reward
            else:
                self.T = self.t # terminal
            
    def do_td_state_value_updates(self, sv_coll, alpha=0.1, gamma=1.0,
                                  start_state_hash=None): # only used for policy, not episode_obj
        """
        Given an initialized NStepTDWalker,
        Iterate through the returns for the episode (either Episode object OR policy episode)
        
        Update the StateValueColl, sv_coll as part of the episode iteration.
        NOTE: policy does NOT change.
        This routine only calculates the StateValueColl FOR the given policy.
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
                    G += g_pow * sv_coll.get_Vs( self.S[ self.tau+self.Nsteps ] )
                    
                delta = alpha * ( G - sv_coll.get_Vs( self.S[ self.tau ] ) )                
                sv_coll.delta_update( s_hash=self.S[ self.tau ], delta=delta )
                #print('t=%i'%self.t, '  tau=%i'%self.tau, '  T=%g'%self.T, '  self.S[t]=%s'%str(self.S[self.t]), '  delta=%g'%delta)
            
    def do_td_sv_function_updates(self, sv_func, alpha=0.1, gamma=1.0,
                                  start_state_hash=None): # only used for policy, not episode_obj
        """
        Given an initialized NStepTDWalker,
        Iterate through the returns for the episode (either Episode object OR policy episode)
        
        Update the Baseline_V_Func, sv_func as part of the episode iteration.
        NOTE: policy does NOT change.
        This routine only calculates the w_vector of Baseline_V_Func FOR the given policy.
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
                if self.S[self.t+1] in self.environment.terminal_set:
                    self.T = self.t + 1
                    
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
                    G += g_pow * sv_func.VsEst( self.S[ self.tau+self.Nsteps ] )
                
                sv_func.mc_update( s_hash=self.S[ self.tau ], alpha=alpha, G=G)
                
    
if __name__ == "__main__":  # pragma: no cover
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    from introrl.policy import Policy
    from introrl.agent_supt.epsilon_calc import EpsilonGreedy
    from introrl.agent_supt.episode_maker import make_episode
    from introrl.agent_supt.state_value_coll import StateValueColl
    
    gridworld = get_gridworld()
    sv = StateValueColl( gridworld )
    
    pi = Policy( environment=gridworld )
    
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    #pi.summ_print()
    
    eg = EpsilonGreedy(epsilon=0.5, const_epsilon=True, half_life=200,
                   N_episodes_wo_decay=0)

    
    episode_obj = make_episode( (2,0), pi, gridworld, eps_greedy=None )
    
    """environment, Nsteps=3, 
                 policy=None, episode_obj=None, 
                 terminal_set=None,
                 max_steps=sys.maxsize, eps_greedy=None"""
    
    print('Using an episode_obj')
    episode_obj.summ_print()
    print('                ...')
    NSW = NStepTDWalker( gridworld, Nsteps=16, episode_obj=episode_obj )
    NSW.do_td_state_value_updates( sv, alpha=0.1, gamma=0.9, start_state_hash=None)
    #print()
    #gridworld.summ_print()
    print()
    sv.summ_print( fmt_V='%g', none_str='*', show_states=True,
                   show_last_change=True )
    
    print('-'*55) # -----------------------------------------------------------------
    # =========================================================================================
    
    print('Simple Policy Following')
    sv = StateValueColl( gridworld )
    NSW = NStepTDWalker( gridworld, Nsteps=3, policy=pi, eps_greedy=None )
    NSW.do_td_state_value_updates( sv, alpha=0.1, gamma=0.9, start_state_hash=(2,0))
    sv.summ_print( fmt_V='%g', none_str='*', show_states=True,
                   show_last_change=True )

    print('-'*55)
    
    
