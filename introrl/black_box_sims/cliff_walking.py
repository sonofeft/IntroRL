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
from introrl.black_box_sims.sim_baseline import Simulation

class CliffWalkingSimulation( Simulation ):
    
    def __init__(self, name='Cliff Walking Sim', width=12, height=4,
                 s_hash_rowL=None,
                 row_tickL=None, col_tickL=None, 
                 x_axis_label='', y_axis_label='',
                 step_reward=-1):
        """
        A Black Box Interface to a Simulation
        """
        self.step_reward = step_reward
        self.width = width
        self.height = height

        # -------- make layout template for states ---------
        s_hash_rowL = [] # layout rows for makeing 2D output
        for i in range(height-1):
            rowL = []
            for j in range(width):
                s_hash = (i,j)
                rowL.append( s_hash )
            # use insert to put (0,0) at lower left, append for upper left
            s_hash_rowL.append(rowL )# layout rows for makeing 2D output
        
        rowL = ['S']
        for j in range(width-2):
            rowL.append( '"Cliff"' )
        rowL.append('G')
        s_hash_rowL.append(rowL )# layout rows for makeing 2D output
        
        # call parent object
        Simulation.__init__(self, name=name, s_hash_rowL=s_hash_rowL)
        
        # state hash
        self.action_state_set = set(['S']) # a set of state hashes
        for s1 in range( height-1 ): # 20 cars max
            for s2 in range( width ): # 20 cars max
                self.action_state_set.add( (s1, s2) )
    
        self.terminal_set = set(['G'])
        
        self.start_state_hash = 'S'
    

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        # make start a special case
        if s_hash == 'S':
            if a_desc == 'U':
                return (self.height-2, 0), self.step_reward
            else:
                return 'S', self.step_reward
        
        (s1, s2) = s_hash
                
        # all other moves
        di = 0
        dj = 0

        if a_desc=='U':
            di = -1
        elif a_desc=='D':
            di = 1
        elif a_desc=='R':
            dj = 1
        elif a_desc=='L':
            dj = -1
        i_next = s1 + di
        
        # constrain basic move to be inside the grid
        i_next = max(0, min(self.height-1, i_next))
        j_next = s2 + dj
        j_next = max(0, min(self.width-1, j_next))

        if i_next==self.height-1:
            if (j_next>0) and (j_next<self.width-1):
                sn_hash = self.start_state_hash # 'Cliff'
                reward = -100
            elif j_next==0:
                sn_hash = 'S'
                reward = self.step_reward
            elif j_next==self.width-1:
                sn_hash = 'G'
                reward = self.step_reward
        else:
            sn_hash = (i_next, j_next)
            reward = self.step_reward
        
        if sn_hash == (self.height-1, 0):
            return 'S', self.step_reward
        
        return sn_hash, reward
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        if s_hash == 'S':
            return ['U']#,'R']
        elif s_hash in ['G']: #,'Cliff']:
            return []
        
        return ['U','D','R','L']

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
    from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
    
    CW = CliffWalkingSimulation()
    CW.layout.s_hash_print( none_str='*' )
    
    policy, state_value = \
        sarsa_epsilon_greedy( CW, 
                              initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                              read_pickle_file='', 
                              save_pickle_file='',
                              use_list_of_start_states=False, # use list OR single start state of environment.
                              do_summ_print=True, show_last_change=True, fmt_Q='%g', fmt_R='%g',
                              max_num_episodes=500, min_num_episodes=10, max_abserr=0.001, gamma=1.0,
                              iteration_prints=0,
                              max_episode_steps=1000,
                              epsilon=0.1, const_epsilon=True, epsilon_half_life=200,
                              alpha=0.1, const_alpha=True, alpha_half_life=200,
                              N_episodes_wo_decay=0)
    
