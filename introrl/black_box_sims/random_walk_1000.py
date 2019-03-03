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

jump_rangeL = list( range(1,101) )

class RandomWalk_1000Simulation( Simulation ):
    
    def __init__(self, name='Random Walk 1000 Sim', 
                 s_hash_rowL=None,
                 row_tickL=None, col_tickL=None, 
                 x_axis_label='', y_axis_label=''):


        # -------- make layout template for states ---------
        # break 1000 states into 40 rows of 25
        s_hash_rowL = [] # layout rows for makeing 2D output
        N = 1
        for i in range(40):
            rowL = []
            for j in range(25):
                rowL.append( N )
                N += 1
            s_hash_rowL.append(rowL )# layout rows for makeing 2D output
                
        # call parent object
        Simulation.__init__(self, name=name, s_hash_rowL=s_hash_rowL)
        
        # state hash
        self.action_state_set = set( list(range(2,1000)) ) # a set of state hashes
    
        self.terminal_set = set([1, 1000])
        
        self.start_state_hash = 500
    

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        
        jump_size = random.choice( jump_rangeL )
        
        if a_desc == 'L':
            snext_hash = s_hash - jump_size
        elif a_desc == 'R':
            snext_hash = s_hash + jump_size
        
        if snext_hash <= 1:
            return 1, -1
        elif snext_hash >= 1000:
            return 1000, 1
        else:
            return snext_hash, 0

        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        if s_hash <= 1:
            return []
        elif s_hash >= 1000:
            return []
        else:
            return ['L','R']
        

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.agent_supt.model import Model
    from introrl.environments.env_baseline import EnvBaseline
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    from introrl.utils import pickle_esp
    
    start_time = time.time()
    
    RW = RandomWalk_1000Simulation()
    #RW.layout.s_hash_print( none_str='*' )
    
    
    get_sim = Model( RW, build_initial_model=True )

    get_sim.collect_transition_data( num_det_calls=100, num_stoic_calls=10000 )

    RW.layout.s_hash_print()

    #get_sim.num_calls_layout_print()
    #get_sim.min_num_calls_layout_print()
    
    env = EnvBaseline( s_hash_rowL=RW.s_hash_rowL, 
                       x_axis_label=RW.x_axis_label, 
                       y_axis_label=RW.y_axis_label )
                       
    get_sim.add_all_data_to_an_environment( env )

    policy, state_value = dp_value_iteration( env, do_summ_print=True, fmt_V='%.3f', fmt_R='%.1f',
                                              max_iter=1000, err_delta=0.0001, 
                                              gamma=0.9, iteration_prints=10)
                                  
    policy.save_diagram( RW, inp_colorD=None, save_name='dp_rw1000_policy',
                         show_arrows=False, scale=0.5, h_over_w=0.8,
                         show_terminal_labels=False)

    print( 'Total Time =',time.time() - start_time )

    pickle_esp.save_to_pickle_file( fname='dp_soln_to_randwalk_1000', 
                                    env=env, state_values=state_value, policy=policy)



