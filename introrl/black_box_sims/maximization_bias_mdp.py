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

class MaximizationBiasMDP( Simulation ):
    
    def __init__(self, name='Maximization Bias MDP Sim', width=12, Nb_choices=10,
                 s_hash_rowL=None,
                 row_tickL=None, col_tickL=None, 
                 x_axis_label='', y_axis_label=''):
        """
        A Black Box Interface to a Simulation
        """
        self.Nb_choices = Nb_choices # number of choices from B

        # -------- make layout template for states ---------
        s_hash_rowL = [['Lterm','B','A','Rterm']] # layout rows for makeing 2D output
        
        # call parent object
        Simulation.__init__(self, name=name, s_hash_rowL=s_hash_rowL)
        
        # state hash is
        self.action_state_set = set(['B','A']) # a set of state hashes
    
        self.terminal_set = set(['Lterm','Rterm'])
        
        self.start_state_hash = 'A'
    

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        # make start a special case
        if s_hash == 'A':
            if a_desc == 'Right':
                return 'Rterm',0.0
            else:
                return 'B', 0.0
        
        # if not in 'A', then must be in 'B'
        sn_hash = 'Lterm'
        reward = random.gauss(-0.1, 1.0)
        
        return sn_hash, reward
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        if s_hash == 'A':
            return ['Left','Right']
        elif s_hash == 'B':
            return list( range(self.Nb_choices) )
            
        return []

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
    from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
    from introrl.agent_supt.episode_maker import make_episode
    from introrl.agent_supt.episode_summ_print import epi_summ_print
    
    MB = MaximizationBiasMDP()
    MB.layout.s_hash_print( none_str='*' )
    
    policy, state_value = \
        qlearning_epsilon_greedy( MB, 
                              initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                              use_list_of_start_states=False, # use list OR single start state of environment.
                              do_summ_print=True, show_last_change=True, fmt_Q='%g', fmt_R='%g',
                              pcent_progress_print=0,
                              show_banner = True,
                              max_num_episodes=10, min_num_episodes=10, max_abserr=0.001, 
                              gamma=1.0,
                              max_episode_steps=100,
                              epsilon=0.1, 
                              alpha=0.1)
    
                          
    episode = make_episode( MB.start_state_hash, policy, MB, 
                            MB.terminal_set, max_steps=20 )
    epi_summ_print(episode, policy, MB, show_rewards=False,
                   show_env_states=True, none_str='*')
                          