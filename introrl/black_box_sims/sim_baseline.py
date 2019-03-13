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
from introrl.layouts.generic_layout import GenericLayout

class Simulation( object ):
    
    def __init__(self, name='Basic Sim', s_hash_rowL=None, 
                 row_tickL=None, col_tickL=None, 
                 x_axis_label='', y_axis_label='', 
                 colorD=None, basic_color='',
                 start_time=0):
        """
        A Black Box Interface to a Simulation
        """
        self.name = name
        
        self.info = """A Black Box Interface to a Simulation."""
        
        self.s_hash_rowL = s_hash_rowL
        self.row_tickL = row_tickL
        self.col_tickL = col_tickL
        
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        
        self.colorD = colorD
        self.basic_color = basic_color
        
        # state hash is (# cars at 1st site, # cars at 2nd site)
        self.action_state_set = set([0,1,2,3]) # a set of action state hashes
        self.terminal_set = set([4]) # a set of terminal state hashes
    
        if s_hash_rowL is None:
            self.layout = None # may have a layout object for display purposes. (e.g. GenericLayout)
        else:
            self.layout = GenericLayout( self, s_hash_rowL=s_hash_rowL, 
                                         row_tickL=row_tickL, col_tickL=col_tickL, 
                                         x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                                         colorD=colorD, basic_color=basic_color)

        self.default_policyD = None # may define later.

    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        if s_hash in self.action_state_set:
            sn_hash = s_hash + a_desc
        else:
            sn_hash = None
        
        if sn_hash == 4:
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
        if s_hash in self.action_state_set:
            return [1]
        else:
            return []

    def limited_start_state_list(self):
        """
        Return a limited list of starting states.
        Normally used by agents that need to discover the various
        states in an environment, like epsilon-greedy.
        
        OVERRIDE THIS to return a list of states smaller than 
        ALL ACTION STATES.
        """
        return list(self.action_state_set)
    
    def get_policy_score(self, policy=None, start_state_hash=None, step_limit=1000):
        """
        Given a Policy object, OR policy dictionary,
        apply it to the Simulation and return a score
        
        Can iterate over limited_start_state_list, or simply start at start_state_hash.
        """
        if policy is None:
            policy = self.default_policyD
        
        if start_state_hash is None:
            try:
                s_hash = self.start_state_hash
            except:
                s_hash = self.limited_start_state_list()[0]
        else:
            s_hash = start_state_hash
            
        r_sum = 0.0
        n_steps = 0
        a_desc = policy.get( s_hash, None)
        
        while (a_desc is not None) and (n_steps<step_limit):
            
            sn_hash, reward = self.get_action_snext_reward( s_hash, a_desc )
            
            try: # if reward is numeric, add to r_sum
                r_sum += reward
            except:
                pass
                
            n_steps += 1
            
            s_hash = sn_hash
            a_desc = policy.get( s_hash, None)
        
            
        msg = '' # any special message(s)
        return (r_sum, n_steps, msg)# can OVERRIDE this to return a more meaningful score.
    
    # ========> The Following Methods Can Simply Be Inherited
        
    def set_info(self, info):
        """Input string that describes Environment."""
        self.info = info
    
    def get_info(self):
        lmax = max( [len(s) for s in self.info.split('\n')] )
        lmax = max( 16, lmax )
        
        return '\n' + 'INFO'.center(lmax, '_') + '\n' + self.info + '\n' + '_'*lmax + '\n'
    
    
    def is_legal_state(self, s_hash):
        """legal if either terminal or not."""
        if s_hash in self.terminal_set:
            return True
        else:
            return s_hash in self.action_state_set
    
    def is_terminal_state(self, s_hash):
        return s_hash in self.terminal_set
    
    def get_num_states(self):
        return len(self.action_state_set) + len(self.terminal_set)
    
    def get_num_action_states(self):
        return len(self.action_state_set)
    
    def get_num_terminal_states(self):
        return len(self.terminal_set)
    
    def iter_all_action_states(self, randomize=False):
        """iterate over all action states in environment"""
        if randomize:
            for s_hash in random.sample( self.action_state_set, len(self.action_state_set) ):
                yield s_hash # assume none in terminal_set
        else:
            for s_hash in self.action_state_set:
                yield s_hash # assume none in terminal_set
    
    def iter_all_terminal_states(self):
        """iterate over all terminal states in environment"""
        for s_hash in self.terminal_set:
            yield s_hash # assume none in action_state_set
    
    def iter_all_states(self):
        """iterate over all states in environment"""
        for s_hash in self.iter_all_action_states():
            yield s_hash # assume none in terminal_set

        for s_hash in self.iter_all_terminal_states():
            yield s_hash # assume none in action_state_set
        
    def get_set_of_all_terminal_state_hashes(self):
        """
        Return a set of terminal state hash values. OR empty set.
        (No non-terminal states should be included.)
        Primarily used to detect the end of an episode.
        """
        return self.terminal_set
        
    def get_all_action_state_hashes(self):
        """
        Return a list of action state hash values. OR empty list.
        (No terminal states should be included.)
        """
        return list(self.action_state_set)
        
    def get_any_action_state_hash(self):
        """
        Return a action state hash.
        Can be the same state every time, some random state,
        some state from a set sequence, anything at all.
        """
        return random.choice( tuple( self.action_state_set ) )

    def get_default_policy_desc_dict(self):
        """
        If the simulation has a default policy, return it as a dictionary
            index=state_hash, value=action_desc
            
        NOTE: for deterministic policy, probability of each action is 1.0
              so do not need to return tuples of (action, probability)
        """
        # Policy Dictionary
        if self.default_policyD is None:
            return {}
        else:
            return self.default_policyD

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    from introrl.environments.env_baseline import EnvBaseline
    #from introrl.black_boxes.collect_sim_data import CollectSimData
    from introrl.agent_supt.model import Model
    
    start_time = time.time()

    s_hash_rowL = ( (0,1,2,3,4), )
    CR = Simulation( s_hash_rowL=s_hash_rowL )
    
    #get_sim = CollectSimData( CR )
    get_sim = Model( CR, build_initial_model=True )
    
    # if there's a pickle file, read it
    fname = os.path.split( __file__ )[-1].split('.')[0] # use file prefix for pickle file
    print('Pickle File Name Prefix:', fname)
    
    if not get_sim.read_pickle_file( fname ):
        get_sim.collect_transition_data( num_det_calls=10, num_stoic_calls=1000 )
    
    print('Total recorded actions Before:', "{:,}".format( get_sim.total_num_action_data_points() ) )    
    get_sim.collect_transition_data( num_det_calls=10, num_stoic_calls=100 )
    print('Total recorded actions After:', "{:,}".format( get_sim.total_num_action_data_points() ) )    
        
    get_sim.save_to_pickle_file( fname )
    
        
    #get_sim.summ_print( long=False )
    print('got sim data')
    print('_'*55)
    
    
    env = EnvBaseline( s_hash_rowL=CR.s_hash_rowL )
    get_sim.add_all_data_to_an_environment( env )
    
    #env.layout = GenericLayout( env )
    
    print('built environment')
    print('_'*55)
    
    #env.summ_print()
    policy, state_value = dp_value_iteration( env, do_summ_print=True, fmt_V='%.1f', fmt_R='%.1f',
                                              max_iter=1000, err_delta=0.0001, 
                                              gamma=0.9, iteration_prints=10)
                                              
    print( 'Total Time =',time.time() - start_time )
        