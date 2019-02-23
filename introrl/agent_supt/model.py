#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import pickle
import os
import random

from introrl.agent_supt.model_state_data import ModelStateData
from introrl.utils.gen_sort_key import NaturalOrStrKey

class Model( object ):
    """
    Build up a model of a environment simulation by interacting with it.
    """
    
    def __init__(self, env_interface, build_initial_model=False): # Interface (can be sim or env)
        
        self.name = env_interface.name
        self.env_interface = env_interface # Interface (can be sim or env)
        
        self.define_statesD = {} # index=s_hash: value=ModelStateData object for s_hash
        self.total_action_calls = 0 # track total number of action calls
        
        if build_initial_model:
            # try to build up basic model on startup by calling env_interface
            # NOTE: Assume terminal states show up as sn_hash references.
            for s_hash in self.env_interface.iter_all_action_states():
                aL = self.env_interface.get_state_legal_action_list( s_hash )
                for a_desc in aL:
                    self.add_action( s_hash, a_desc )
                    
                    sn_hash, rwd = self.env_interface.get_action_snext_reward( s_hash, a_desc )
                    
                    self.save_action_results( s_hash, a_desc, sn_hash, reward_val=rwd )
    
    def get_random_state(self):
        if self.define_statesD:
            return random.choice( tuple( self.define_statesD.keys() ) )
        else:
            return None
        
    def get_random_action(self, s_hash):
        """
        Select a random action from s_hash.
        """
        
        if s_hash in self.define_statesD:
            model_data = self.define_statesD[s_hash]
            return random.choice( tuple( model_data.action_countD.keys() ) )
        else:
            #print(s_hash,'not in self.define_statesD =',self.define_statesD)
            return None

    def get_sample_sn_r(self, s_hash, a_desc):
        
        if s_hash in self.define_statesD:
            model_data = self.define_statesD[s_hash]
            return model_data.get_sample_sn_r( a_desc )
        else:
            return None, None

    def get_ave_reward(self, s_hash, a_desc):
        
        if s_hash in self.define_statesD:
            model_data = self.define_statesD[s_hash]
            return model_data.get_ave_reward( a_desc )
        else:
            return 0

    def get_ave_reward_to_snext(self, s_hash, a_desc, sn_hash_inp):
        """If stochastic sn_hash, this will only look at sn_hash_inp"""        
        if s_hash in self.define_statesD:
            model_data = self.define_statesD[s_hash]
            return model_data.get_ave_reward_to_snext( a_desc, sn_hash_inp )
        else:
            return 0
    
    def total_num_action_data_points(self):
        """Add up all the calls to get_action_snext_reward."""
        Ntotal = 0
        for s_hash, model_data in self.define_statesD.items():
            Ntotal += model_data.total_action_calls
        return Ntotal
    
    def make_pickle_filename(self, fname):
        """Make a file name ending with .bb_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.bb_pickle'
            
        else:
            fname = fname.replace(' ','_').replace('.','_') + '.bb_pickle'
            
        return fname
    
    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        
        fname = self.make_pickle_filename( fname )
        
        saveD = {}
        saveD['name'] = self.name
        saveD['define_statesD'] = self.define_statesD
        
        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
    
    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle file."""
        
        fname = self.make_pickle_filename( fname )
        if not os.path.isfile( fname ):
            print('Pickle File NOT found:', fname)
            return False
        
        fileObject = open(fname,'rb')  
        readD = pickle.load(fileObject)  
        
        self.name = readD['name']
        self.define_statesD = readD['define_statesD']
        
        fileObject.close()
        return True

    def has_state(self, s_hash):
        """Return True if s_hash is already in the model"""
        return s_hash in self.define_statesD

    def has_sa(self, s_hash, a_desc):
        """Return True if (s_hash, a_desc) is already in the model"""
        if s_hash in self.define_statesD:
            return self.define_statesD[ s_hash ].has_action( a_desc )
        else:
            return False
                
    def add_action(self, s_hash, a_desc):
        if s_hash not in self.define_statesD:
            self.define_statesD[ s_hash ] = ModelStateData( s_hash )
        
        self.define_statesD[ s_hash ].add_action( a_desc )
        
        self.total_action_calls += 1
        
    def save_action_results(self, s_hash, a_desc, snext_hash, reward_val=0.0):
        self.define_statesD[ s_hash ].save_action_results( a_desc, snext_hash,  reward_val )

    def save_deterministic_action_results(self, s_hash, a_desc, snext_hash, reward_val=0.0):
        """Save results such that any previous different snext_hash is overritten."""
        self.define_statesD[ s_hash ].save_action_results( a_desc, snext_hash,  reward_val, 
                                                           force_deterministic=True )

    def add_all_data_to_an_environment(self, env):
        """Apply all the collected env_interface data to the Environment variable, env."""    
        for s_hash, rsa in self.define_statesD.items():
            rsa.add_to_environment( env )
        
        # Environment states need update
        env.terminal_set, env.action_state_set = env.TC.get_terminal_set_and_action_set()
        
        # if possible, refresh layout info.
        # shouldn't be necessary... layout should have been defined on env creation
            
    def collect_transition_data(self, num_det_calls=10, num_stoic_calls=1000):
        """
        If (s,a) pairs appear deterministic after num_calls, 
        assume they are deterministic
        """
        print('Collecting Simulation Data for:', self.name)
        print('     Number of Deterministic Calls=%i, Stochastic Calls=%i'%(num_det_calls, num_stoic_calls) )
        def get_N_transitions(N, rsa, s_hash ):
            for _ in range( N ):
                for a_desc in rsa.action_countD.keys():
                    sn_hash, rwd = self.env_interface.get_action_snext_reward( s_hash, a_desc )
                    self.save_action_results( s_hash, a_desc, sn_hash, reward_val=rwd )

        for s_hash, rsa in self.define_statesD.items():
            # only worry about states "claiming" to be deterministic
            if rsa.all_state_actions_deterministic():
                num_calls = num_det_calls
                assume_det = True
            else:
                num_calls = num_stoic_calls
                assume_det = False
            
            get_N_transitions(num_calls, rsa, s_hash )
            
            if assume_det:
                # after the above calls, if no longer deterministic make stoichastic calls
                if not rsa.all_state_actions_deterministic():
                    num_calls = max(0, num_stoic_calls - num_det_calls)
                    if num_calls > 0:
                        get_N_transitions(num_calls, rsa, s_hash )

    def num_calls_layout_print(self, row_tickL=None, const_col_w=True,
                               header='Total (s,a) Calls', x_axis_label='', none_str='*'):
        """print a layout of the total number of calls to each (s,a) pair"""
        num_callsD = {}
        for s_hash, model_data in self.define_statesD.items():
            num_callsD[s_hash] = str(model_data.total_action_calls)
            #print('num_callsD[s_hash]',s_hash,num_callsD[s_hash])
        #print('self.define_statesD.items()',self.define_statesD.items())
            
        self.env_interface.layout.param_print(num_callsD, row_tickL=row_tickL, const_col_w=const_col_w,
                                             header=header, x_axis_label=x_axis_label, none_str=none_str)

    def min_num_calls_layout_print(self, row_tickL=None, const_col_w=True,
                               header='Calls to Least Frequent Action', x_axis_label='', none_str='*'):
        """print a layout of the least freqent actions."""
        num_callsD = {}
        for s_hash, model_data in self.define_statesD.items():
            min_calls = model_data.total_action_calls
            for a_desc, snrD in model_data.action_sn_rD.items():
                for sn_hash, R in snrD.items():
                    min_calls = min( min_calls, R.num_val )
            
            num_callsD[s_hash] = str( min_calls )
            
        self.env_interface.layout.param_print(num_callsD, row_tickL=row_tickL, const_col_w=const_col_w,
                                             header=header, x_axis_label=x_axis_label, none_str=none_str)


    def est_reward_error_layout_print(self, row_tickL=None, const_col_w=True,
                                      header='Estimated Reward Error', x_axis_label='', none_str='*'):
        """Print a layout of estimated rewards."""
        err_rangeD = {}
        for s_hash, model_data in self.define_statesD.items():
            pc_err_max=0.0
            pc_err_min=100.0
            
            for a_desc, snrD in model_data.action_sn_rD.items():
                if model_data.is_deterministic_action( a_desc ):
                    pc_err_min = 0.0
                else:
                    for sn_hash, R in snrD.items():
                        pc_err = R.get_est_pcent_err()
                        pc_err_max = max(pc_err, pc_err_max)
                        pc_err_min = min(pc_err, pc_err_min)
                    
            if pc_err_max < 1.0E-10:
                err_rangeD[s_hash] = '0.0%'
            else:
                #err_rangeD[s_hash] = '%.1f-%.1f%%'%(pc_err_min, pc_err_max)
                err_rangeD[s_hash] = '%.1f%%'%pc_err_max
            
        self.env_interface.layout.param_print(err_rangeD, row_tickL=row_tickL, const_col_w=const_col_w,
                                             header=header, x_axis_label=x_axis_label, none_str=none_str)

    def summ_print(self, long=False): # pragma: no cover
        header = 'Model Data: %s'%str(self.name)
        print( header )
        
        if long:
            sL = sorted( [s_hash for s_hash in self.define_statesD.keys()], key=NaturalOrStrKey )
            for s_hash in sL:
                RSA = self.define_statesD[s_hash]
                RSA.summ_print()
        
if __name__ == "__main__": # pragma: no cover
    
    from introrl.environments.env_baseline import EnvBaseline
    from introrl.mdp_data.simple_grid_world import get_gridworld

    gridworld = get_gridworld()
    
    get_sim = Model( gridworld, build_initial_model=True ) # <-- DynaQ uses False for build_initial_model
    if 1:
        get_sim.collect_transition_data( num_det_calls=10, num_stoic_calls=1000 )
        get_sim.save_to_pickle_file( 'temp' )
    else:
        get_sim.read_pickle_file( 'temp' )
        
    get_sim.summ_print( long=True )
    
    print('_'*55)
    env = EnvBaseline()
    get_sim.add_all_data_to_an_environment( env )
    
    if 0:
        print('_'*55)
        
        env.summ_print()
        
    
                