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

from math import exp, factorial

MAX_CARS = 20
N1_LAMBDA = 3
N2_LAMBDA = 4

N1_RTNS = 3
N2_RTNS = 2

RENTAL_CREDIT = 10
MOVE_CAR_COST = 2

# -------- make layout template for states ---------

s_hash_rowL = [] # layout rows for makeing 2D output
for s1 in range( MAX_CARS + 1 ): # 20 cars max
    rowL = [] # row of s_hash_rowL
    for s2 in range( MAX_CARS + 1 ): # 20 cars max
        s_hash = (s1, s2)
        rowL.append( s_hash )
    # use insert to put (0,0) at lower left
    s_hash_rowL.insert(0, rowL )# layout rows for makeing 2D output



def knuth_poisson(lam):
    """
    Return a random poisson value for the given lambda value.
    Knuth algorithm from https://en.wikipedia.org/wiki/Poisson_distribution
    """
    elam = exp(-lam)
    k = 0
    p = 1.0
    
    while True:
        k += 1
        u = random.random()
        p *= u
        if p <= elam:
            return k-1

class CarRentalSimulation( Simulation ):
    
    def __init__(self, name='Car Rental Sim Const Rtns', s_hash_rowL=s_hash_rowL, 
                 x_axis_label='#Cars at Second Location',
                 y_axis_label='#Cars at First Location'):
        """
        A Black Box Interface to a Simulation
        """
        Simulation.__init__(self, name=name, s_hash_rowL=s_hash_rowL)
        
        # state hash is (# cars at 1st site, # cars at 2nd site)
        self.action_state_set = set() # a set of state hashes
        for s1 in range( MAX_CARS + 1 ): # 20 cars max
            for s2 in range( MAX_CARS + 1 ): # 20 cars max
                self.action_state_set.add( (s1, s2) )
    
        self.terminal_set = set()
    

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        (s1, s2) = s_hash
        
        n1 = int(min(s1 - a_desc, MAX_CARS))
        n2 = int(min(s2 + a_desc, MAX_CARS))
        
        n1_rent_request = knuth_poisson( N1_LAMBDA )
        n2_rent_request = knuth_poisson( N2_LAMBDA )
        
        actual_n1_rented = min(n1, n1_rent_request)
        actual_n2_rented = min(n2, n2_rent_request)
        
        reward = (actual_n1_rented + actual_n2_rented) * RENTAL_CREDIT \
               - abs(a_desc) * MOVE_CAR_COST
               
        # next state
        sn1 = int(min(n1 - actual_n1_rented + N1_RTNS, MAX_CARS))
        sn2 = int(min(n2 - actual_n2_rented + N2_RTNS, MAX_CARS))
        sn_hash = (sn1, sn2)

        return sn_hash, reward
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        (s1, s2) = s_hash
        
        # -5 moves 5 cars from 2nd to 1st. +5 from 1st to 2nd.
        a_min = max(-5, -s2) # can only move available cars
        a_max = min(5, s1)
        
        return list( range(a_min, a_max+1 ) )

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    from introrl.environments.env_baseline import EnvBaseline
    from introrl.agent_supt.model import Model
    from introrl.utils import pickle_esp
    
    start_time = time.time()
    
    CR = CarRentalSimulation()
    
    get_sim = Model( CR, build_initial_model=True )
    
    get_sim.collect_transition_data( num_det_calls=50, num_stoic_calls=100000 )
    
    print('Total recorded actions Before:', "{:,}".format( get_sim.total_num_action_data_points() ) )  

    CR.layout.s_hash_print()
    get_sim.num_calls_layout_print(row_tickL=[c for c in '   First Location'], const_col_w=True,
                                   x_axis_label='Second Location', none_str='*')

    get_sim.min_num_calls_layout_print( row_tickL=[c for c in '   First Location'], const_col_w=True,
                                        x_axis_label='Second Location', none_str='*')

    #get_sim.est_reward_error_layout_print(row_tickL=[c for c in '   First Location'], const_col_w=True,
    #                                      x_axis_label='Second Location', none_str='*')

    #get_sim.define_statesD[(20,0)].summ_print()
    
    #sys.exit() # <-------------------------------------
    #get_sim.collect_transition_data( num_det_calls=10, num_stoic_calls=100 )
    #print('Total recorded actions After:', "{:,}".format( get_sim.total_num_action_data_points() ) )    
        
    #get_sim.save_to_pickle_file( fname )
    
        
    #get_sim.summ_print( long=False )
    print('got sim data')
    print('_'*55)
    
    #print('CR.s_hash_rowL =', CR.s_hash_rowL)
    env = EnvBaseline( s_hash_rowL=CR.s_hash_rowL, 
                       x_axis_label=CR.x_axis_label, 
                       y_axis_label=CR.y_axis_label )
                       
    get_sim.add_all_data_to_an_environment( env )

    #env.save_to_pickle_file('car_rental')
    #print('Saved env to *.env_pickle file')
    
    print('built environment')
    print('_'*55)
    
    #env.summ_print()
    policy, state_value = dp_value_iteration( env, do_summ_print=True, fmt_V='%.1f', fmt_R='%.1f',
                                              max_iter=1000, err_delta=0.0001, 
                                              gamma=0.9, iteration_prints=10)
                                              
    print( 'Total Time =',time.time() - start_time )
                
    #env.save_to_pickle_file('car_rental')
    
    pickle_esp.save_to_pickle_file( fname='car_rental_sim_to_env_const_rtn', 
                                    env=env, state_values=state_value, policy=policy)

    