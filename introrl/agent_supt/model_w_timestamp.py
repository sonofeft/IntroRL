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
from math import sqrt

from introrl.agent_supt.model_state_data import ModelStateData
from introrl.utils.gen_sort_key import NaturalOrStrKey
from introrl.utils.functions import argmax_vmax_list
from introrl.agent_supt.model import Model

class ModelWTimestamp( Model ):
    """
    Build up a model of a environment simulation by interacting with it.
    """
    
    def __init__(self, env_interface, build_initial_model=False): # Interface (can be sim or env)
        
        # add dictionary to track time_stamp
        self.state_action_time_stampD = {} # index=(s_hash,a_desc), value=time_stamp
        
        Model.__init__(self, env_interface, build_initial_model=build_initial_model )
    
    def get_qplus_reward_bonus(self, s_hash, a_desc, qplus_factor=0.0, time_stamp=0):
        """
        If time_stamp>0 and qplus_factor>0.0, use DynaQ+ logic to calc reward bonus
        """
        r_bonus = 0.0
        if (s_hash in self.define_statesD) and ((s_hash, a_desc) in self.state_action_time_stampD):
            if (time_stamp>0) and (qplus_factor > 0.0):
                #delta_time = time_stamp - self.state_action_time_stampD.get( (s_hash, a_desc), 0 )
                delta_time = time_stamp - self.state_action_time_stampD[ (s_hash, a_desc) ]
                if delta_time > 0:
                    r_bonus = qplus_factor * sqrt(delta_time)
    
        return r_bonus
    
    def make_pickle_filename(self, fname):
        """Make a file name ending with .bbt_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.bbt_pickle'
            
        else:
            fname = fname.replace(' ','_').replace('.','_') + '.bbt_pickle'
            
        return fname
    
    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        
        fname = self.make_pickle_filename( fname )
        
        saveD = {}
        saveD['name'] = self.name
        saveD['define_statesD'] = self.define_statesD
        saveD['state_action_time_stampD'] = self.state_action_time_stampD
        
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
        self.state_action_time_stampD = readD['state_action_time_stampD']
        
        fileObject.close()
        return True
    
    def save_action_results(self, s_hash, a_desc, snext_hash, reward_val=0.0, time_stamp=0):
        self.define_statesD[ s_hash ].save_action_results( a_desc, snext_hash,  reward_val )
        
        self.state_action_time_stampD[ (s_hash, a_desc) ] = time_stamp

    def save_deterministic_action_results(self, s_hash, a_desc, snext_hash, 
                                          reward_val=0.0, time_stamp=0):
        """Save results such that any previous different snext_hash is overritten."""
        self.define_statesD[ s_hash ].save_action_results( a_desc, snext_hash,  reward_val, 
                                                           force_deterministic=True )
        self.state_action_time_stampD[ (s_hash, a_desc) ] = time_stamp

    def set_state_action_time_stamp(self, s_hash, a_desc, time_stamp):
        self.state_action_time_stampD[ (s_hash, a_desc) ] = time_stamp

    def summ_print(self, long=False, time_stamp=None): # pragma: no cover
        
        Model.summ_print(self, long=long)
        
        if time_stamp is None:
            """approximate time_stamp with largest model time_stamp"""
            time_stamp = 0
            for t in self.state_action_time_stampD.values():
                time_stamp = max(t, time_stamp)
                
        
        # get all states and figure out formatting
        sL = sorted( [s_hash for s_hash in self.define_statesD.keys()], key=NaturalOrStrKey )
        max_len = max(6, max([len( str(s) ) for s in sL]))
        fmt = '%' + '%is'%max_len
        
        # get all actions for each state and figure out formatting
        astrL = [RSA.get_action_desc_str() for RSA in self.define_statesD.values()]
        max_a_len = max(6, max([len( str(a) ) for a in astrL]))
        fmt_a = '%' + '%is'%max_a_len

        max_a2_len = 0
        max_det_len = 0
        for s_hash in sL:
            RSA = self.define_statesD[s_hash]
            aL = [a_desc for a_desc in RSA.action_countD.keys()]
            max_a2_len = max(max_a2_len, max( [len(a) for a in aL] ))
            
            max_det_len = max(max_det_len, len(RSA.get_state_deterministic_desc().strip()) )
            
        fmt_a2 = '%' + '%is'%max_a2_len
        fmt_det ='%-' + '%is'%max_det_len
                        
        print('___________________________________________________')
        print('             State/Action TimeStamps               ')
        print('___________________________________________________')
        for s_hash in sL:
            RSA = self.define_statesD[s_hash]
            
            aL = sorted( [a_desc for a_desc in RSA.action_countD.keys()], key=NaturalOrStrKey )
            #print('aL =',aL, type(aL))
            
            # self.state_action_time_stampD = {} # index=(s_hash,a_desc), value=time_stamp
            tstampL =  [ fmt_a2%str(a)+'=%i'%(time_stamp - self.state_action_time_stampD[(s_hash,a)],) for a in aL ]
            
            print( fmt%str(s_hash), fmt_a%RSA.get_action_desc_str(),
                   '...', fmt_det%RSA.get_state_deterministic_desc().strip(),' Age:', ', '.join(tstampL) )
            
if __name__ == "__main__": # pragma: no cover
    
    from introrl.environments.env_baseline import EnvBaseline
    from introrl.mdp_data.simple_grid_world import get_gridworld

    gridworld = get_gridworld()
    
    get_sim = ModelWTimestamp( gridworld, build_initial_model=True ) # <-- DynaQ uses False for build_initial_model
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
        
    
                