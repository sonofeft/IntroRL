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
import random

from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.utils.running_ave import RunningAve

class StateValueRunAveColl( object ):
    
    def __init__(self, environment):
        """
        A Collection of RunningAve State-Value, V(s) for each state in the environment
        
        Each Assignment to a value simply updates the running average
        
        To get value use:
        sv.get_ave( s_hash ):
        
        To update value use:
        sv.add_val(s_hash, val)
        
        (Terminal States have V(s) = 0.0)
        """
        
        self.environment = environment
        
        self.Vs_RaveD = {} # index=state_hash, value=RunningAve object
        
        self.init_Vs_to_zero()

    def build_Qsa_from_Vs(self):
        """Build a ActionValueRunAveColl object from this StateValueRunAveColl"""
        pass

    def set_running_ave(self, s_hash, Rave):
        """Set the RunningAve object for a state_hash."""
        self.Vs_RaveD[ s_hash ] = Rave

    def num_Vs(self):
        return len( self.Vs_RaveD )
    
    def init_Vs_to_zero(self):
        # initialize to 0.0 for all states, terminal and non-terminal.
        for s_hash in self.environment.iter_all_states():
            self.Vs_RaveD[ s_hash ] = RunningAve( name=s_hash )
        
    def add_val(self, s_hash, val):
        """add a value to list of returns(G) to calc average  V(s) """
        if s_hash in self.Vs_RaveD:
            self.Vs_RaveD[ s_hash ].add_val( val )
        else:
            raise ValueError( 'No "%s" StateValueRunAveColl exists.'%str(s_hash) )
        
    def get_ave(self, s_hash):
        """Return the average State-Value for s_hash"""
        return self.Vs_RaveD[ s_hash ].get_ave() # Allow key error
    
    def get_biggest_action_state_err(self):
        abserr = float('-inf')
        for s_hash in self.environment.iter_all_action_states():
            if s_hash not in self.environment.terminal_set:
                RA = self.Vs_RaveD[ s_hash ]
                abserr = max(abserr, RA.get_error_estimate())
        return abserr
    
    def make_pickle_filename(self, fname):
        """Make a file name ending with .svra_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.svra_pickle'
            
        else:
            fname = fname.replace(' ','_').replace('.','_') + '.svra_pickle'
            
        return fname
    
    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        
        fname = self.make_pickle_filename( fname )
        
        saveD = {}
        saveD['Vs_RaveD'] = self.Vs_RaveD
        
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
        
        self.Vs_RaveD = readD['Vs_RaveD']
        
        fileObject.close()
        return True
        
    
    def summ_print(self, fmt_V='%g', none_str='*', show_states=True,
                   showRunningAve=True ):
        print()
        print('___ "%s" State-Value Summary ___'%self.environment.name  )
        
        if self.environment.layout is not None:
            # make summ_print using environment.layout
            if show_states:
                self.environment.layout.s_hash_print( none_str='*' )
                
            row_tickL = self.environment.layout.row_tickL
            col_tickL = self.environment.layout.col_tickL
            x_axis_label = self.environment.layout.x_axis_label
            y_axis_label = self.environment.layout.y_axis_label
            
            rows_outL = []
            for row in self.environment.layout.s_hash_rowL:
                outL = []
                for s_hash in row:
                    if not self.environment.is_legal_state( s_hash ):
                        #outL.append( none_str )
                        if is_literal_str( s_hash ):
                            outL.append( s_hash[1:-1] )
                        else:
                            outL.append( none_str )
                    else:
                        outL.append( fmt_V%self.Vs_RaveD[ s_hash ].get_ave() )
                rows_outL.append( outL )
            
            print_string_rows( rows_outL, row_tickL=row_tickL, const_col_w=True,
                               line_chr='_', left_pad='    ', col_tickL=col_tickL,
                               header=self.environment.name + ' State-Value Summary, V(s)', 
                               x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                               justify='right')
            
        
        # ------------------------- simple output w/o a layout ------------
        else:
            lmax_hash = 6
            
            outL = [] # list of tuples = (s_hash, V)
            for s_hash,V in self.Vs_RaveD.items():
                outL.append( (s_hash, V) )
                lmax_hash = max(lmax_hash, len(str(s_hash)))
            fmt_hash = '%' + '%is'%lmax_hash
                    
            outL.sort() # sort in-place
            for (s_hash,  V) in outL:
                print('    ', fmt_hash%str(s_hash), fmt_V%V )
            
        if showRunningAve:
            for s_hash,RA in self.Vs_RaveD.items(): # index=state_hash, value=RunningAve object
                RA.summ_print()
            
if __name__ == "__main__": # pragma: no cover
    
    from introrl.policy import Policy
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    policyD = gridworld.get_default_policy_desc_dict()
    
    pi = Policy( environment=gridworld )
    #pi.learn_all_states_and_actions_from_env( gridworld )
    pi.set_policy_from_piD( policyD )
    
    # -------------
    
    sv = StateValueRunAveColl( gridworld )
    
    sv.add_val((0,0), 2.0)
    sv.add_val((0,0), 3.0)
    print('Value at (0,0) is:', sv.get_ave( (0,0) ) )
    
    sv.summ_print( fmt_V='%6g' )
    
