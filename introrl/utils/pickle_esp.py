#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
import os
import pickle

"""
Save 3 linked objects, environment, state_values, policy
"""

def make_pickle_filename( fname):
    """Make a file name ending with .esp_pickle """
    
    fname = fname.replace(' ','_').replace('.','_') + '.esp_pickle'
        
    return fname

def save_to_pickle_file( fname='env_state_policy', env=None, state_values=None, policy=None):
    """Saves data to pickle file."""
    
    fname = make_pickle_filename( fname )
        
    print('Saving Environment to pickle file:', fname)
    saveD = {}
    saveD['env'] = env
    saveD['state_values'] = state_values
    saveD['policy'] = policy
    
    fileObject = open(fname,'wb')
    pickle.dump(saveD,fileObject)   
    fileObject.close()

def read_pickle_file( fname='env_state_policy'):
    """Reads data from pickle"""
    
    fname = make_pickle_filename( fname )
    if os.path.isfile( fname ):
        pass # all good
    elif os.path.isfile( os.path.join( mdp_path, fname ) ):
        fname = os.path.join( mdp_path, fname )
    else:
        print('Pickle File NOT found:', fname)
        return False, False, False
    
    fileObject = open(fname,'rb')  
    
    readD = pickle.load(fileObject)  
    
    env = readD['env']
    state_values = readD['state_values']
    policy = readD['policy']
    
    fileObject.close()
    
    return env, state_values, policy



if __name__ == "__main__": # pragma: no cover
    
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    from introrl.mdp_data.sample_gridworld import get_gridworld

    if not os.path.isfile( 'sample_gridworld.esp_pickle' ):
        gridworld = get_gridworld()

        policy, state_values = dp_value_iteration( gridworld, do_summ_print=True,
                                                  max_iter=1000, err_delta=0.001, 
                                                  gamma=0.9)
        
        save_to_pickle_file( fname='sample_gridworld', env=gridworld, 
                             state_values=state_values, policy=policy)
    else:
        gridworld, state_values, policy = read_pickle_file( fname='sample_gridworld' )
        gridworld.summ_print()
        print('-'*66)        
        state_values.summ_print()
        print('-'*66)
        policy.summ_print()
    
