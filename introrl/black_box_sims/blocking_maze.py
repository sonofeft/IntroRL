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

class BlockingMaze( Simulation ):
    
    def close_gate_L(self): # Left Gate
        self.Lgate_is_open = False
    
    def close_gate_R(self): # Right Gate
        self.Rgate_is_open = False
    
    def open_gate_L(self):
        self.Lgate_is_open = True
    
    def open_gate_R(self):
        self.Rgate_is_open = True
    
    def __init__(self, name='Blocking Maze Sim', step_reward=0.0, 
                 width=9, height=6, goal=(0,8), start=(5,3), wall_row=3,
                 row_tickL=None, col_tickL=None, 
                 x_axis_label='', y_axis_label=''):
        """
        A Black Box Interface to a Simulation
        Blocking Maze changes route to goal state as gates are opened and closed.
        Starts with Right Gate Open
        """
        self.step_reward = step_reward
        
        self.width = width
        self.height = height
        self.goal = goal
        self.start = start
        self.wall_row = wall_row # far right open at time=0, far left is closed
        
        self.s_hash_gate_L = (wall_row, 0)
        self.s_hash_gate_R = (wall_row, width-1)
        
        self.Lgate_is_open = False
        self.Rgate_is_open = True

        # -------- make layout template for states ---------
        s_hash_rowL = [] # layout rows for makeing 2D output
        for i in range(height):
            rowL = []
            for j in range(width):
                if i==wall_row:
                    if j==0:
                        s_hash = 'Gate_L'
                    elif j==width-1:
                        s_hash = 'Gate_R'
                    else:
                        s_hash = '"Wall"'
                else:
                    if (i,j)==self.goal:
                        s_hash = 'Goal'
                    elif (i,j)==self.start:
                        s_hash = 'Start'
                    else:
                        s_hash = (i,j)
                rowL.append( s_hash )
            # use insert to put (0,0) at lower left, append for upper left
            s_hash_rowL.append(rowL )# layout rows for makeing 2D output
        
        
        # call parent object
        Simulation.__init__(self,  name=name, s_hash_rowL=s_hash_rowL, 
                            row_tickL=row_tickL, col_tickL=col_tickL, 
                            x_axis_label=x_axis_label, y_axis_label=y_axis_label)
        
        # state hash of states with actions.
        self.action_state_set = set() # a list of state hashes
        for s1 in range( height ):
            for s2 in range( width ):
                if s1==wall_row:
                    if s2==0:
                        s_hash = 'Gate_L'
                        self.action_state_set.add( s_hash )
                    elif s2==width-1:
                        s_hash = 'Gate_R'
                        self.action_state_set.add( s_hash )
                else:
                    s_hash = (s1, s2)
                    if s_hash == self.start:
                        s_hash = 'Start'
                    elif s_hash == self.goal:
                        s_hash = 'Goal'
                    
                    if s_hash != 'Goal':
                        self.action_state_set.add( s_hash )
    
        self.terminal_set = set(['Goal'])
        
        self.start_state_hash = 'Start'
            

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        
        if s_hash=='Gate_L':
            (s1, s2) = self.s_hash_gate_L
        elif s_hash=='Gate_R':
            (s1, s2) = self.s_hash_gate_R
        elif s_hash=='Start':
            (s1, s2) = self.start
        elif s_hash=='Goal':
            (s1, s2) = self.goal
        else:
            (s1, s2) = s_hash
                
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
        j_next = s2 + dj
        
        sn_hash = (i_next, j_next)
        if sn_hash == self.s_hash_gate_L:
            sn_hash = 'Gate_L'
            if not self.Lgate_is_open:
                sn_hash = s_hash
        elif sn_hash == self.s_hash_gate_R:
            sn_hash = 'Gate_R'
            if not self.Rgate_is_open:
                sn_hash = s_hash
        elif sn_hash == self.start:
            sn_hash = 'Start'
        elif sn_hash == self.goal:
            sn_hash = 'Goal'
        
        # constrain move to have a legal destination
        if not self.is_legal_state( sn_hash ):
            sn_hash = s_hash
        
        if sn_hash == 'Goal':
            reward = 1.0
        else:
            reward = 0.0
        
        return sn_hash, reward
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        if  s_hash == self.goal:
            return []
        
        return ['U','D','R','L']

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
    from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
    from introrl.agent_supt.model import Model
    
    bmaze = BlockingMaze()
    bmaze.open_gate_R()
    bmaze.close_gate_L()
    
    env = Model( bmaze,  build_initial_model=True)
    env.collect_transition_data( num_det_calls=10, num_stoic_calls=1000 )
    env.summ_print(long=False)
    
    bmaze.layout.s_hash_print( none_str='*' )
    bmaze.open_gate_L()
    bmaze.close_gate_R()
    env.collect_transition_data( num_det_calls=10, num_stoic_calls=1000 )
    env.summ_print(long=False)
    
    policy, action_value = \
        sarsa_epsilon_greedy( bmaze, 
                              initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                              read_pickle_file='', 
                              save_pickle_file='',
                              use_list_of_start_states=False, # use list OR single start state of environment.
                              do_summ_print=True, show_last_change=True, fmt_Q='%g', fmt_R='%g',
                              show_banner = True,
                              max_num_episodes=5000, min_num_episodes=10, max_abserr=0.001, gamma=0.95,
                              iteration_prints=0,
                              max_episode_steps=1000,
                              epsilon=0.1, const_epsilon=True, epsilon_half_life=200,
                              alpha=0.1, const_alpha=True, alpha_half_life=200,
                              N_episodes_wo_decay=0)
    
