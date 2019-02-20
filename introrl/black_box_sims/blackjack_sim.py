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
from introrl.black_box_sims.blackjack_supt.blackjack import  BlackJack, play_policy_game # ( BJ, policyD )


# -------- make layout template for states ---------
s_hash_rowL = [] # layout rows for makeing 2D output
row_tickL = []
col_tickL = []

for usable_ace in (False, True):
    if usable_ace:
        low_lim = 12
    else:
        low_lim = 11
    
    for player_sum in list(range(low_lim, 22, 1)) + ['Win','Draw','Lose']:
        rowL = [] # row of s_hash_rowL
        row_tickL.insert(0, str(player_sum) )
        
        for dealer_showing in range(1, 11, 1):
            s_hash = (player_sum, usable_ace, dealer_showing)
            rowL.append( s_hash )
        
        # use insert to put (0,0) at lower left
        s_hash_rowL.insert(0, rowL )# layout rows for makeing 2D output
    
    #if usable_ace:
    #    #s_hash_rowL.insert(0, ['*']*len(rowL) )# extra rows to separate usable ace from not
    #    #s_hash_rowL.insert(0, ['*']*len(rowL) )# extra rows to separate usable ace from not
    #    #row_tickL.insert(0, '' )
    #    #row_tickL.insert(0, '' )

# make column tick labels
for dealer_showing in range(1, 11, 1):
    col_tickL.append( dealer_showing )


class BlackJackSimulation( Simulation ):
    
    def __init__(self, name='BlackJack Simulation', s_hash_rowL=s_hash_rowL, 
                 row_tickL=row_tickL, 
                 col_tickL=col_tickL, 
                 x_axis_label='Dealer Showing', y_axis_label='Usable Ace   Player Sum    No Usable Ace'):
        """
        A Black Box Interface to a Simulation
        """
        Simulation.__init__(self, name=name, s_hash_rowL=s_hash_rowL, 
                            row_tickL=row_tickL, x_axis_label=x_axis_label,
                            y_axis_label=y_axis_label, col_tickL=col_tickL)
        
        self.default_policyD = {} # will define as Hit on everything except 20 or 21
        
        # state hash is (# cars at 1st site, # cars at 2nd site)
        self.action_state_set = set() # a set of action state hashes
        for usable_ace in (True, False):
            if usable_ace:
                low_lim = 12
            else:
                low_lim = 11
            
            for player_sum in range(low_lim, 22, 1):
                for dealer_showing in range(1, 11, 1):
                    s_hash = (player_sum, usable_ace, dealer_showing)
                    self.action_state_set.add( s_hash )
                    
                    if player_sum < 20:
                        self.default_policyD[ s_hash ] = 'Hit'
                    else:
                        self.default_policyD[ s_hash ] = 'S'

        terminalL = [] # terminal state hashes.
        for usable_ace in (True, False):
            for player_sum in ['Win','Draw','Lose']:
                for dealer_showing in range(1, 11, 1):
                    s_hash = (player_sum, usable_ace, dealer_showing)
                    terminalL.append( s_hash )


        self.terminal_set = set( terminalL )
        
        self.bj_hand = BlackJack()
    

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        
        self.bj_hand.set_state_hash( s_hash )
        
        (player_sum, usable_ace, dealer_showing) = s_hash
        
        if a_desc == 'Hit':
            self.bj_hand.player_hits()
            if self.bj_hand.playerH.is_bust:
                sn_hash = ('Lose', usable_ace, dealer_showing)
            else:
                sn_hash = self.bj_hand.get_state_hash()
                
        else: # if not Hit, then Stay
            self.bj_hand.dealer_plays()
            
            if self.bj_hand.reward > 0.1:
                sn_hash = ('Win', usable_ace, dealer_showing)
            elif self.bj_hand.reward < -0.1:
                sn_hash = ('Lose', usable_ace, dealer_showing)
            else:
                sn_hash = ('Draw', usable_ace, dealer_showing)
            
            
        return sn_hash, self.bj_hand.reward
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        (player_sum, usable_ace, dealer_showing) = s_hash
        
        try:
            if player_sum < 21:
                return ['Hit','S']
            else:
                return ['S']
        except:
            return []

    def limited_start_state_list(self):
        """
        Return a limited list of starting states.
        Normally used by agents that need to discover the various
        states in an environment, like epsilon-greedy.
        """
        
        lim_stateL = []
        for dealer_showing in range(1, 11, 1):
            for player_sum in range(12, 22, 1):
                s_hash = (player_sum, True, dealer_showing)
                lim_stateL.append( s_hash )
            for player_sum in range(11, 22, 1):
                s_hash = (player_sum, False, dealer_showing)
                lim_stateL.append( s_hash )

        return lim_stateL

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    from introrl.environments.env_baseline import EnvBaseline
    from introrl.agent_supt.model import Model
    
    start_time = time.time()
    
    BJ = BlackJackSimulation()
    
    get_sim = Model( BJ, build_initial_model=True )
    
    # if there's a pickle file, read it
    fname = os.path.split( __file__ )[-1].split('.')[0] # use file prefix for pickle file
    print('Pickle File Name Prefix:', fname)
    
    if not get_sim.read_pickle_file( fname ):
        get_sim.collect_transition_data( num_det_calls=10, num_stoic_calls=10000 )

    #get_sim.collect_transition_data( num_det_calls=10, num_stoic_calls=10000 )

    print('Total recorded actions Before:', "{:,}".format( get_sim.total_num_action_data_points() ) )  

    BJ.layout.s_hash_print()
    
    get_sim.num_calls_layout_print()#row_tickL=[c for c in '   Player Sum'], const_col_w=True,
                                   #x_axis_label='Dealer Showing', none_str='*')

    get_sim.min_num_calls_layout_print()# row_tickL=[c for c in '   Player Sum'], const_col_w=True,
                                        #x_axis_label='Dealer Showing', none_str='*')

    get_sim.est_reward_error_layout_print()#row_tickL=[c for c in '   Player Sum'], const_col_w=True,
                                          #x_axis_label='Dealer Showing', none_str='*')

    
    #sys.exit() # <-------------------------------------
    get_sim.collect_transition_data( num_det_calls=10, num_stoic_calls=100 )
    print('Total recorded actions After:', "{:,}".format( get_sim.total_num_action_data_points() ) )    
        
    get_sim.save_to_pickle_file( fname )
    
        
    #get_sim.summ_print( long=False )
    print('got sim data')
    print('_'*55)
    
    #print('BJ.s_hash_rowL =', BJ.s_hash_rowL)
    
    env = EnvBaseline( s_hash_rowL=BJ.s_hash_rowL, 
                       row_tickL=BJ.row_tickL, col_tickL=BJ.col_tickL,
                       x_axis_label=BJ.x_axis_label, y_axis_label=BJ.y_axis_label )
                       
    get_sim.add_all_data_to_an_environment( env )
    
    print('built environment')
    print('_'*55)
    
    #env.summ_print()
    policy, state_value = dp_value_iteration( env, do_summ_print=True, fmt_V='%.2f', fmt_R='%.2f',
                                              max_iter=1000, err_delta=0.0001, 
                                              gamma=0.9, iteration_prints=10)
                                              
    print( 'Total Time =',time.time() - start_time )
    
    env.save_to_pickle_file('blackjack_env')
                
