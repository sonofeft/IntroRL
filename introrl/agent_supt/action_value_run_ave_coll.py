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
import random
import pickle

from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.utils.running_ave import RunningAve
from introrl.agent_supt.state_value_run_ave_coll import StateValueRunAveColl

class ActionValueRunAveColl( object ):
    
    def __init__(self, environment):
        """
        A Collection of RunningAve Action-Value, Q(s,a) 
        for each state-action pair in the environment
        
        Each Assignment to a value simply updates the running average
        
        To get value use:
        qsa.get_ave( s_hash, a_desc ):
        
        To update value use:
        qsa.add_val(s_hash, a_desc, val)
        """
        
        self.environment = environment
        
        self.Qsa_RaveD = {} # index=s_hash value=aD (dict),  aD index=a_desc, value=RunningAve object
        
        self.init_Qsa_to_zero()

    def build_Vs_from_Qsa(self, environment):
        """
        Build a StateValueRunAveColl object from this ActionValueRunAveColl
        i.e. choose max action Q(s,a) for each V(s)
        """
        sv = StateValueRunAveColl( environment )
        for s_hash, aD in self.Qsa_RaveD.items():
            best_val = float('-inf')
            Rbest = None
            for a_desc, Rave in aD.items():
                if Rave.get_ave() > best_val:
                    best_val = Rave.get_ave()
                    Rbest = Rave
        
            if Rbest is not None:
                Rnew = Rbest.clone( s_hash )
                sv.set_running_ave(s_hash, Rnew)
        
        return sv

    def build_policy_Vs_from_Qsa(self, policy):
        """
        For the given policy,
        build a StateValueRunAveColl object from this ActionValueRunAveColl
        i.e. will be Vpi(s) NOT V*(s)
        """
        pass

    def num_Qsa(self):
        return len( self.Qsa_RaveD )
    
    def init_Qsa_to_zero(self):
        # initialize to 0.0 for all states, terminal and non-terminal.
        for s_hash in self.environment.iter_all_states():
            if s_hash not in self.Qsa_RaveD:
                self.Qsa_RaveD[s_hash] = {}

            # may not be any actions in terminal state, so set None action.
            if s_hash in self.environment.terminal_set:
                self.Qsa_RaveD[s_hash][None] = RunningAve( name=str(s_hash) + ' None' )
                
            aL = self.environment.get_state_legal_action_list( s_hash )
            for a_desc in aL:
                self.Qsa_RaveD[s_hash][a_desc] = RunningAve( name=str(s_hash) + ' ' + str(a_desc) )
        
    def add_val(self, s_hash, a_desc, val):
        """add a value to list of returns(G) to calc average  Q(s,a) """
        if s_hash in self.Qsa_RaveD and a_desc in self.Qsa_RaveD[s_hash]:
            self.Qsa_RaveD[s_hash][a_desc].add_val( val )
        else:
            raise ValueError( 'No "%s" ActionValueRunAveColl exists.'%str((s_hash, a_desc)) )
        
    def get_ave(self, s_hash, a_desc):
        """Return the average Action-Value for (s_hash, a_desc)"""
        return self.Qsa_RaveD[s_hash][a_desc].get_ave() # Allow key error
    
    def get_biggest_action_state_err(self):
        abserr = float('-inf')
        for s_hash in self.Qsa_RaveD.keys():
            if s_hash not in self.environment.terminal_set:
                for RA in self.Qsa_RaveD[s_hash].values():
                    abserr = max(abserr, RA.get_error_estimate())
        return abserr
    
    def make_pickle_filename(self, fname):
        """Make a file name ending with .vave_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.vave_pickle'
            
        else:
            fname = fname.replace(' ','_').replace('.','_') + '.vave_pickle'
            
        return fname
    
    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        # build name for pickle
        fname = self.make_pickle_filename( fname )
        
        saveD = {}
        saveD['Qsa_RaveD'] = self.Qsa_RaveD
        
        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
        print('Saved ActionValueRunAveColl to file:',fname)
    
    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle file."""
        
        fname = self.make_pickle_filename( fname )
        if not os.path.isfile( fname ):
            print('Pickle File NOT found:', fname)
            return False
        
        fileObject = open(fname,'rb')  
        readD = pickle.load(fileObject)  
        
        Qsa_RaveD = readD['Qsa_RaveD']
        
        fileObject.close()
        print('Read ActionValueRunAveColl from file:',fname)
        
        return Qsa_RaveD

    def init_from_pickle_file(self, fname=None): # pragma: no cover
        """Initialize ActionValueRunAveColl from policy pickle file."""
        Qsa_RaveD = self.read_pickle_file( fname=fname )
        if Qsa_RaveD:
            self.Qsa_RaveD = Qsa_RaveD
            
    def summ_print(self, fmt_Q='%g', none_str='*', show_states=True,
                   showRunningAve=True ):
        print()
        print('___ "%s" Action-Value Summary ___'%self.environment.name  )
        
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
                        #aL = self.environment.get_state_legal_action_list( s_hash )
                        aD = self.Qsa_RaveD[s_hash]
                        sL = [str(s_hash)]
                        for a_desc, Q in aD.items():
                            s = fmt_Q%Q.get_ave()
                            sL.append( '%s='%str(a_desc) + s.strip()  )
                        outL.append(  '\n'.join(sL).strip()  )
                rows_outL.append( outL )
            
            print_string_rows( rows_outL, row_tickL=row_tickL, const_col_w=True,
                               line_chr='_', left_pad='    ', col_tickL=col_tickL,
                               header=self.environment.name + ' Action-Value Summary, Q(s,a)', 
                               x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                               justify='right')
            
        
        # ------------------------- simple output w/o a layout ------------
        else:
            lmax_hash = 6
            
            outL = [] # list of strings "(s_hash,a_desc)=Q"
            for s_hash in self.Qsa_RaveD.keys():
                for  a_desc,Q in self.Qsa_RaveD[s_hash].items():
                    q = fmt_Q%Q.get_ave()
                    s = '(%s, %s)='%(str(s_hash),str(a_desc)) + q.strip()
                    outL.append( s )
                    lmax_hash = max(lmax_hash, len(s))
                outL.sort() # sort in-place
            for s in outL:
                print('    ', s )
            
            
if __name__ == "__main__": # pragma: no cover
    
    from introrl.policy import Policy
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    policyD = gridworld.get_default_policy_desc_dict()
    
    pi = Policy( environment=gridworld )
    #pi.learn_all_states_and_actions_from_env( gridworld )
    pi.set_policy_from_piD( policyD )
    
    # -------------
    
    av = ActionValueRunAveColl( gridworld )
    
    av.add_val((0,0),'R', 2.0)
    av.add_val((0,0),'D', 3.0)
    print('Value at ((0,0),"R") is:', av.get_ave( (0,0),"R" ) )
    
    #gridworld.layout = None
    
    av.summ_print( fmt_Q='%6g' )
    print('-'*55)
    gridworld.layout = None
    av.summ_print( fmt_Q='%6g' )
    
