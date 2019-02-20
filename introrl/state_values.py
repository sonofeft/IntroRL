#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import os, sys
import random
import pickle

from introrl.utils.grid_funcs import print_string_rows

class StateValues( object ):
    
    def __init__(self, environment):
        """
        State-Value, V(s) for each state in the environment
        (Terminal States have V(s) = 0.0)
        Uses s_id to identify each state.
        """
        
        self.environment = environment
        
        # self.Vs_hashD[s_hash] = v
        self.VsD = {} # index=state_hash, value=estimated State-Value (float)
        
        if environment is not None:
            self.init_Vs_to_zero()
    
    def init_Vs_to_zero(self):
        # initialize to 0.0 for all states, terminal and non-terminal.
        #for s_hash in self.environment.SC.iter_state_hash():
        for s_hash in self.environment.iter_all_states():
            self.VsD[ s_hash ] = 0.0
            
    def __call__(self, s_hash):
        """ 
        Enable access to values with form: 
        sv(s_hash) 
        """
        return self.VsD[ s_hash ] # Allow key error
        
    def __setitem__(self, s_hash, val):
        """ 
        Enable setting values with form: 
        sv[s_hash] = val
        """
        if s_hash in self.VsD:
            self.VsD[ s_hash ] = val
        else:
            raise ValueError( 'No "%s" StateValue exists.'%str(s_hash) )
    
    def make_pickle_filename(self, fname):
        """Make a file name ending with .vs_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.vs_pickle'
            
        else:
            fname = fname.replace(' ','_').replace('.','_') + '.vs_pickle'
            
        return fname
    
    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        
        fname = self.make_pickle_filename( fname )
        
        saveD = {}
        saveD['VsD'] = self.VsD
        
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
        
        self.VsD = readD['VsD']
        #print('VsD',self.VsD)
        
        fileObject.close()
        return True
        

    def summ_print(self, fmt_V='%g', none_str='*', show_states=True):
        print()
        print('___ "%s" State-Value Summary ___'%self.environment.name  )
        
        if self.environment.layout is not None:
            # make summ_print using environment.layout
            if show_states:
                self.environment.layout.s_hash_print( none_str='*' )
            
            
            rows_outL = []
            for row in self.environment.layout.s_hash_rowL:
                outL = []
                for s_hash in row:
                    if s_hash not in self.environment.SC.stateD:
                        outL.append( none_str )
                    else:
                        outL.append( fmt_V%self.VsD[ s_hash ] )
                rows_outL.append( outL )
            
            print_string_rows( rows_outL, row_tickL=self.environment.layout.row_tickL, 
                               col_tickL=self.environment.layout.col_tickL, 
                               const_col_w=True,
                               line_chr='_', left_pad='    ', 
                               header=self.environment.name + ' State-Value Summary, V(s)', 
                               y_axis_label=self.environment.layout.y_axis_label,
                               x_axis_label=self.environment.layout.x_axis_label, justify='right')
            
        
        # ------------------------- simple output w/o a layout ------------
        else:
            lmax_hash = 6
            
            outL = [] # list of tuples = (s_hash, V)
            for s_hash,V in self.VsD.items():
                outL.append( (s_hash, V) )
                lmax_hash = max(lmax_hash, len(str(s_hash)))
            fmt_hash = '%' + '%is'%lmax_hash
                    
            outL.sort() # sort in-place
            for (s_hash,  V) in outL:
                print('    ', fmt_hash%str(s_hash), fmt_V%V )
            

if __name__ == "__main__": # pragma: no cover
    
    from introrl.policy import Policy
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    policyD = gridworld.get_default_policy_desc_dict()
    
    pi = Policy( environment=gridworld )
    #pi.learn_all_states_and_actions_from_env( gridworld )
    pi.set_policy_from_piD( policyD )
    
    # -------------
    
    sv = StateValues( gridworld )
    
    sv.summ_print( fmt_V='%6g' )
    
