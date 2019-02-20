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

# define layout to create output displays
row_1 = [ (0,0), (0,1),   (0,2), 'Goal' ]
row_2 = [ (1,0),'"Wall"', (1,2), 'Pit' ]
row_3 = [ 'Start', (2,1),   (2,2), (2,3) ]
s_hash_rowL=[row_1, row_2, row_3]

# add layout row and column markings (if any)
row_tickL=[ 0, 1, 2]
col_tickL=[ 0, 1, 2, 3]
x_axis_label='cols'
y_axis_label='rows'

# one way to define actions is an explicit dict of actions.
# (can also simply provide logic within a function to define actions)
actionD = {(0, 0): ('D', 'R'),
           (0, 1): ('L', 'R'),
           (0, 2): ('L', 'D', 'R'),
           (1, 0): ('U', 'D'),
           (1, 2): ('U', 'D', 'R'),
           'Start': ('U', 'R'),
           (2, 1): ('L', 'R'),
           (2, 2): ('L', 'R', 'U'),
           (2, 3): ('L', 'U')  }

# define rewards
rewardD = {'Goal': 1, 'Pit': -1}


class SampleSimulation( Simulation ):
    
    def __init__(self, name='Sample Gridworld Sim',
                 step_reward=-0.04, 
                 random_transition_prob=0.2):
                     
        """A Simulation of a Sample Gridworld"""
        
        self.step_reward = step_reward
        
        # probability of moving in random direction.
        self.random_transition_prob = random_transition_prob
        
        # call parent object
        Simulation.__init__(self, name=name, 
                            s_hash_rowL=s_hash_rowL,
                            row_tickL=row_tickL, 
                            col_tickL=col_tickL, 
                            x_axis_label=x_axis_label, 
                            y_axis_label=y_axis_label)

        
        # state hash is
        self.action_state_set = set( actionD.keys() ) # a set of state hashes
    
        self.terminal_set = set( rewardD.keys() )

        # if there is a start state, define it.
        self.start_state_hash = 'Start'

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return next state, sn_hash, and reward
        """
        # default is 80% take input a_desc, 20% choose randomly
        if random.random() < self.random_transition_prob:
            a_desc = random.choice( actionD[s_hash] )
    
        # put 'Start' into (row,col) form
        if s_hash == 'Start':
            s_hash = (2,0)
            
        row,col = s_hash # all non-terminal s_hash are (row, col)
        if a_desc == 'U':
            row -= 1
        elif a_desc == 'D':
            row += 1
        elif a_desc == 'R':
            col += 1
        elif a_desc == 'L':
            col -= 1
        # no limit checking done... assume only legal moves are submitted
        sn_hash = s_hash_rowL[row][col]

        reward = rewardD.get(sn_hash, self.step_reward)
        
        return sn_hash, reward
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        
        return actionD.get( s_hash, [] )
        

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.agent_supt.episode_maker import make_episode
    from introrl.agent_supt.episode_summ_print import epi_summ_print
    from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
    
    
    sim = SampleSimulation( step_reward=-0.04 )
    sim.layout.s_hash_print( none_str='*' )
    
    
    policy, state_value = \
        sarsa_epsilon_greedy( sim, 
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
    
                          
    episode = make_episode( sim.start_state_hash, policy, sim, 
                            sim.terminal_set, max_steps=20 )
    epi_summ_print(episode, policy, sim, show_rewards=False,
                   show_env_states=True, none_str='*')

    sim.random_transition_prob = 0.0 # so arrows are drawn deterministically on policy diagram
    policy.save_diagram( sim, inp_colorD=None, save_name='sample_sim_policy',
                         show_arrows=True, scale=1.0, h_over_w=0.8, do_show=True)

