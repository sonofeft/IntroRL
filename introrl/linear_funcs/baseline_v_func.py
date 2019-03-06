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
import numpy as np
import random
import pickle
from introrl.agent_supt.change_tracker import ChangeTracker
from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.policy import Policy


class Baseline_V_Func( object ):
    """
    Create a linear function for an environment that simply one-hot encodes
    all of the states.
    
    OVERRIDE THIS for more interesting linear functions.
    
    This is only interesting for debugging linear function solution routines.
    (i.e. each term in the one-hot encoding should move to near the actual 
    value function)
    """
    
    # ======================== OVERRIDE STARTING HERE ==========================
    def init_w_vector(self):
        """Initialize the weights vector and the number of entries, N."""
        
        # initialize a weights numpy array with random values.
        N = len(self.sD)
        self.w_vector = np.random.randn(N) / np.sqrt(N)
        self.N = len( self.w_vector )
                
    def get_x_vector(self, s_hash ):
        """Return the x vector that represents the (s,a) pair."""
        x_vector = np.zeros(self.N, dtype=np.float)
        x_vector[ self.sD[ s_hash ] ] = 1.0
        return x_vector
    # ======================== OVERRIDE ENDING HERE ==========================
    
    def __init__(self, environment):
        
        self.environment = environment
        
        self.chgTracker = ChangeTracker()
        self.init_tracking()
        
        # initialize a weights numpy array with random values.
        self.init_w_vector()
        # e.g.  self.w_vector = np.random.randn(self.N) / np.sqrt(self.N)
        self.N = len(self.w_vector)

    def init_tracking(self):
        # initialize known states.
        self.sD = {}
        self.last_delta_VsD = {}  # index=s_hash value=last change to s_hash
        
        # initialize to init_val for all states, terminal = 0.0
        for s_hash in self.environment.iter_all_states():
            # set dict value to index of numpy array
            self.sD[ s_hash ] = len(self.sD)
            self.last_delta_VsD[s_hash] = 0.0
                
    def get_number_of_changes(self):
        return self.chgTracker.get_number_of_changes()

    def num_Vs(self):
        return len( self.sD )

    def record_changes(self, s_hash, delta ):
        """Keep track of changes made to V(s) values"""
        
        delta = abs(delta) # make sure that only absolute values are saved.
        
        # remove any record of last change to [s_hash]
        self.chgTracker.dec_change( self.last_delta_VsD[s_hash] )
        
        # add delta to tracking record
        self.chgTracker.inc_change( delta )
        
        # remember that delta was last change to  [s_hash]
        self.last_delta_VsD[s_hash] = delta
    
    def get_biggest_action_state_err(self):
        """Estimate the biggest error in all the action values."""
        #print('self.chgTracker.get_biggest_change()', self.chgTracker.get_biggest_change())
        return self.chgTracker.get_biggest_change()

    def get_max_last_delta_overall(self):
        """ get biggest entry in self.last_delta_VsD # index=s_hash value=aD (dict)"""
        d_max = 0.0
        for aD in self.last_delta_VsD.values():
            for val in aD.values():
                d_max = max(d_max, abs(val))
        return d_max

    def VsEst(self, s_hash):
        """Return the current estimate for V(s) from linear function eval."""
        x_vector = self.get_x_vector( s_hash )
        return self.w_vector.dot( x_vector )
    
    def get_gradient(self, s_hash):
        """
        Return the gradient of value function with respect to w_vector.
        Since the function is linear in w, the gradient is = x_vector.
        """
        return self.get_x_vector( s_hash )

    def mc_update(self, s_hash='', alpha=0.1, G=0.0):
        """
        Do a Monte-Carlo-style learning rate update.
        w = w + alpha * [Gt - Vhat(st)] * grad(st)
        """
        Vs    = self.VsEst( s_hash )
        delta = alpha * (G - Vs)
        
        delta_vector = delta * self.get_gradient( s_hash )
        self.w_vector += delta_vector
        
        delta = np.max( np.absolute( delta_vector ) )
        self.record_changes( s_hash, delta )
        
        return abs(delta) # return the absolute value of change

    def td0_update(self, s_hash='', alpha=0.1, gamma=1.0, sn_hash='', reward=0.0):
        """
        Do a TD(0), Temporal-Difference-style learning rate update.
        w = w + alpha * [R + gamma*VEst(s',w) - V(s,w)] * grad(s)
        """
        Vs    = self.VsEst( s_hash )
        
        if sn_hash in self.environment.terminal_set:
            target_val = reward
        else:
            Vstp1 = self.VsEst( sn_hash )
            target_val = reward + gamma*Vstp1
            
        delta = alpha * (target_val - Vs)
        
        delta_vector = delta * self.get_gradient( s_hash )
        self.w_vector += delta_vector
        
        delta = np.max( np.absolute( delta_vector ) )
        self.record_changes( s_hash, delta )
        
        return abs(delta) # return the absolute value of change
        

    # ========================== pickle routines ===============================

    def make_pickle_filename(self, fname):
        """Make a file name ending with .vlf_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.vlf_pickle'

        else:
            fname = fname.replace(' ','_').replace('.','_') + '.vlf_pickle'

        return fname

    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        # build name for pickle
        fname = self.make_pickle_filename( fname )

        saveD = {}
        saveD['sD'] = self.sD
        saveD['last_delta_VsD'] = self.last_delta_VsD
        saveD['w_vector'] = self.w_vector

        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
        print('Saved ActionValueColl to file:',fname)

    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle file."""

        fname = self.make_pickle_filename( fname )
        if not os.path.isfile( fname ):
            print('Pickle File NOT found:', fname)
            return False

        fileObject = open(fname,'rb')
        readD = pickle.load(fileObject)

        sD = readD['sD']
        last_delta_VsD = readD['last_delta_VsD']
        w_vector = readD['w_vector']

        fileObject.close()
        print('Read ActionValueColl from file:',fname)

        return sD, last_delta_VsD, w_vector

    def init_from_pickle_file(self, fname=None): # pragma: no cover
        """Initialize ActionValueColl from policy pickle file."""
        sD, last_delta_VsD, w_vector = self.read_pickle_file( fname=fname )
        if sD:
            self.sD = sD
            self.w_vector = w_vector
            self.last_delta_VsD = last_delta_VsD
            self.N = len(self.w_vector)
            self.chgTracker = ChangeTracker()
            self.init_tracking()
        else:
            print('ERROR... Failed to read file:', fname)


    # ========================== summ_print ===============================

    
    def summ_print(self, fmt_V='%g', none_str='*', show_states=True,
                   show_last_change=True ):
        print()
        print('___ "%s" Alpha-Based State-Value Summary ___'%self.environment.name  )
        
        if self.environment.layout is not None:
            # make summ_print using environment.layout
            if show_states:
                self.environment.layout.s_hash_print( none_str='*' )
                
            row_tickL = self.environment.layout.row_tickL
            col_tickL = self.environment.layout.col_tickL
            x_axis_label = self.environment.layout.x_axis_label
            y_axis_label = self.environment.layout.y_axis_label
            
            rows_outL = []
            last_delta_rows_outL = [] # if show_last_change == True
            for row in self.environment.layout.s_hash_rowL:
                outL = []
                ld_outL = []
                ld_outL.append( none_str )
                for s_hash in row:
                    if not self.environment.is_legal_state( s_hash ):
                        if is_literal_str( s_hash ):
                            outL.append( s_hash[1:-1] )
                            ld_outL.append( s_hash[1:-1] )
                        else:
                            outL.append( none_str )
                            ld_outL.append( none_str )
                    else:
                        outL.append( fmt_V%self.VsEst( s_hash ) )
                        delta = self.last_delta_VsD.get(s_hash, None)
                        if delta is None:
                            ld_outL.append( 'None' )
                        else:
                            ld_outL.append( fmt_V%delta )
                        
                rows_outL.append( outL )
                last_delta_rows_outL.append( ld_outL )
            
            print_string_rows( rows_outL, row_tickL=row_tickL, const_col_w=True,
                               line_chr='_', left_pad='    ', col_tickL=col_tickL,
                               header=self.environment.name + ' State-Value Summary, V(s)', 
                               x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                               justify='right')
            if show_last_change:
                print_string_rows( last_delta_rows_outL, row_tickL=row_tickL, const_col_w=True,
                                   line_chr='_', left_pad='    ', col_tickL=col_tickL,
                                   header=self.environment.name + ' Last Change to V(s) Summary', 
                                   x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                                   justify='right')
            
        # ------------------------- simple output w/o a layout ------------
        else:
            lmax_hash = 6
            lmax_V = 6
            
            outL = [] # list of tuples = (s_hash, V)
            for s_hash,_ in self.VsD.items():
                V = self.VsEst( s_hash )
                outL.append( (s_hash, V) )
                
                lmax_hash = max(lmax_hash, len(str(s_hash)))
                lmax_V = max(lmax_V, len(fmt_V%V) )
                
            fmt_hash = '%' + '%is'%lmax_hash
            fmt_strV = '%' + '%is'%lmax_V
                    
            outL.sort() # sort in-place
            for (s_hash,  V) in outL:
                V = fmt_V%V
                print('    ', fmt_hash%str(s_hash), fmt_strV%V, end='' )
                if show_last_change:
                    print( ' Last Delta = %s'%self.last_delta_VsD.get(s_hash, None) )
                else:
                    print()
        

if __name__ == "__main__": # pragma: no cover
    import sys
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()

    oh = Baseline_V_Func( gridworld )
    
    SAVE_MODE = 1
    if SAVE_MODE:
        for i in range( oh.N ):
            oh.w_vector[i] = 0.1
    else:
        oh.init_from_pickle_file( fname='testing_lf_save')
    
    # -------------------------------
    old_w_vector = oh.w_vector.copy()
    
    
    oh.td0_update( s_hash=(0,0),  alpha=0.1, gamma=1.0,
                   sn_hash=(0,1), reward=-1.0)
    
    oh.td0_update( s_hash=(0,2),  alpha=0.1, gamma=1.0,
                   sn_hash=(0,3), reward=1.0)
    
    oh.mc_update(s_hash=(0,2), alpha=0.1, G=1.0)
    
    #oh.environment.layout = None
    oh.summ_print()
    #sys.exit()
                     
    print('w_vector')
    for s_hash in gridworld.iter_all_states():
        i = oh.sD[ s_hash ]
        if old_w_vector[i] == oh.w_vector[i]:
            print(s_hash, '%.5f'%old_w_vector[i], '---> Both Equal')
        else:
            print(s_hash, '%.5f'%old_w_vector[i], '%.5f'%oh.w_vector[i])
    
    if SAVE_MODE:
        oh.save_to_pickle_file( fname='testing_lf_save')
    
    print('='*66)
    print('oh.chgTracker:')
    oh.chgTracker.summ_print()
    print()
    print('Biggest a,s error =', oh.get_biggest_action_state_err() )
    
    #sys.exit()
    
    for _ in range(10):
        oh.record_changes( (2,random.choice([0,1,2])), random.random() )
    print('oh.chgTracker:')
    oh.chgTracker.summ_print()
    print()
    print('Biggest a,s error =', oh.get_biggest_action_state_err() )
    print()
    print('='*66)
    
    print('x_vector')
    for s_hash in gridworld.iter_all_states():
        x_vector = oh.get_x_vector( s_hash )
        print(s_hash, '[', ' '.join(['%g'%v for v in x_vector]), ']')
    print('='*66)
    print('vs_est')
    for s_hash in gridworld.iter_all_states():
        vs = oh.VsEst( s_hash )
        print(s_hash, vs)
    print('='*66)
    print('gradient')
    for s_hash in gridworld.iter_all_states():
        gradient = oh.get_gradient( s_hash )
        print(s_hash, '[', ' '.join(['%g'%v for v in gradient]), ']')
    