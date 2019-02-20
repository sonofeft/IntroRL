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
import random
from math import sqrt
import copy

from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.agent_supt.change_tracker import ChangeTracker
from introrl.agent_supt.model_state_data import ModelStateData
from introrl.utils.functions import argmax_vmax_dict
from introrl.policy import Policy

class StateValueColl( object ):
    
    def __init__(self, environment, init_val=0.0):
        """
        A Collection of State-Value, V(s) for each state in the environment
        
        Each Update to V(s) is done with a learning-rate (alpha)
        
        To get value use:
        sv.get_Vs( s_hash ):
        
        To update value use:
        sv.delta_update( s_hash, delta)
        sv.mc_update( s_hash, alpha, G)
        sv.td0_update( s_hash, alpha, gamma, sn_hash, reward)
        etc.
        
        (Terminal States have V(s) = 0.0)
        
        Transition probabilities from (s,a) to (sn,reward) are collected
        as updates happen.
        """
        
        self.environment = environment
        
        # check to see if the environment already contains trasition probabilities
        if hasattr( environment, 'iter_next_state_prob_reward'):
            self.env_has_transition_prob = True
        else:
            self.env_has_transition_prob = False
        
        self.VsD = {} # index=state_hash, value=state value, Vs (a float)
        
        self.define_statesD = {} # index=s_hash: value=ModelStateData object for s_hash
        
        # (used in error estimate)
        # Monte Carlo = Gt, discounted return
        # TD(0) = Rt+1 + gamma*V(st+1) (estimated discounted return)
        
        self.last_delta_VsD = {} # index=s_hash value=last change to s_hash
        self.chgTracker = ChangeTracker()
        
        self.init_Vs_to_val( init_val )
        self.init_val = init_val
        
        self.min_target = None # initialize when 1st target is submitted
        self.max_target = None

    def num_Vs(self):
        return len( self.VsD )
    
    def init_Vs_to_val(self, init_val):
        # initialize to init_val for all states, terminal = 0.0
        for s_hash in self.environment.iter_all_states():
            self.last_delta_VsD[ s_hash ] = 0.0 # record last change as 0.0
            
            if s_hash in self.environment.terminal_set:
                self.VsD[ s_hash ] = 0.0
            else:
                self.VsD[ s_hash ] = init_val

    def get_best_env_action(self, s_hash, a_descL ):
        """Given env_has_transition_prob == True, find best action from given list."""
        
        VsD = {} # will hold: index=a_desc, value=V(s) for all transitions of a from s
        
        # iterate over all actions from s, MUST include zero prob actions
        for a_desc in a_descL:
            calcd_v = 0.0
            # iterate over the probability of going to next state, sn when action, a  is taken
            for sn_hash, t_prob, reward in \
                self.environment.iter_next_state_prob_reward(s_hash, a_desc, incl_zero_prob=False):
                
                # use probability-averaged V(sn) values from state_value_coll
                calcd_v += t_prob * (reward + self.VsD[ sn_hash ])
        
            VsD[a_desc] = calcd_v
        
        best_a, best_a_val = argmax_vmax_dict( VsD )
        return best_a, best_a_val
    
    def get_best_blackbox_action(self, s_hash, a_descL ):
        """Given env_has_transition_prob == False, find best action from given list."""
        
        if s_hash in self.define_statesD: # index=s_hash: value=ModelStateData object for s_hash
            
            VsD = {} # will hold: index=a_desc, value=V(s) for all transitions of a from s
            PD = self.define_statesD[s_hash]
            
            for a_desc in a_descL:
                # select any actions not yet taken to start getting trasition data
                if a_desc not in PD.action_sn_rD:
                    return a_desc, self.init_val # <--- Jumps the line to return unused actions.
                
                # if the action is deterministic (so far), just look up the current V(s)
                if PD.is_deterministic_action( a_desc ):
                    snD = PD.action_sn_rD[ a_desc ]
                    sn_hash = tuple( snD.keys() )[0]
                    rwd_ave_obj = snD[ sn_hash ]
                    VsD[a_desc] = rwd_ave_obj.get_ave() + self.VsD[ sn_hash ]
                else:
                    # for stochastic actions, do a transition probability weighted calc of V(s)
                    calcd_v = 0.0
                    a_count = PD.action_countD.get( a_desc, 0 )# index=a_desc: value=count of (s,a) occurances
                    if a_count > 0:
                        
                        if a_desc in PD.action_sn_rD:
                            snD = PD.action_sn_rD[ a_desc ] # snD...  index=sn_hash: value=rwd_ave_obj
                            for sn_hash, rwd_ave_obj in snD.items():
                    
                                # fraction of times using a_desc in s_hash resulted in sn_hash
                                t_prob = float(rwd_ave_obj.num_val) / float(a_count)
                                calcd_v += t_prob * ( rwd_ave_obj.get_ave() + self.VsD[ sn_hash ])
                
                    VsD[a_desc] = calcd_v
            
            best_a, best_a_val = argmax_vmax_dict( VsD )
            return best_a, best_a_val
            
        else:
            # this state has not yet been called so initialize transition tracking for it.
            for a_desc in a_descL:
                self.add_action( s_hash, a_desc )
            
            return a_desc, self.init_val

    def get_best_eps_greedy_action(self, s_hash, epsgreedy_obj=None ):
        """
        Pick the best action for state "s_hash" based on max V(s')
        If epsgreedy_obj is given, apply Epsilon Greedy logic to choice.
        """
        a_descL = self.environment.get_state_legal_action_list( s_hash )
        if a_descL:
            if self.env_has_transition_prob:
                best_a_desc, best_a_val = self.get_best_env_action( s_hash, a_descL )
            else:
                best_a_desc, best_a_val = self.get_best_blackbox_action( s_hash, a_descL )
                    
            if epsgreedy_obj is not None:
                best_a_desc = epsgreedy_obj( best_a_desc, a_descL )
                    
            return best_a_desc
        return None
        
            
    def record_changes(self, s_hash, delta ):
        """Keep track of changes made to V(s) values"""
        
        delta = abs(delta) # make sure that only absolute values are saved.
        
        # remove any record of last change to [s_hash]
        self.chgTracker.dec_change( self.last_delta_VsD[s_hash] )
        
        # add delta to tracking record
        self.chgTracker.inc_change( delta )
        
        # remember that delta was last change to  [s_hash]
        self.last_delta_VsD[s_hash] = delta


    def get_snapshot(self):
        """
        return a deep copy of the value dictionary.
        index=state_hash, value=state value, Vs (a float)
        """
        return copy.deepcopy( self.VsD )

    def delta_update(self, s_hash='', delta=0.0):
        """Add delta to current value of s_hash"""
        self.VsD[ s_hash ] += delta
        
        self.record_changes( s_hash, delta )
                
    def add_action(self, s_hash, a_desc):
        """
        Add an action to trasition data with call as follows.
        self.add_action( s_hash, a_desc )
        """
        if s_hash not in self.define_statesD:
            self.define_statesD[ s_hash ] = ModelStateData( s_hash )
            self.define_statesD[ s_hash ].add_action( a_desc )
        
    def save_action_results(self, s_hash, a_desc, sn_hash, reward_val):
        """Add sn_hash to possible next states and add to its RunningAve"""
        
        self.add_action( s_hash, a_desc )
        self.define_statesD[ s_hash ].save_action_results( a_desc, sn_hash, reward_val )

    def mc_update(self, s_hash='', alpha=0.1, G=0.0):
        """
        Do a Monte-Carlo-style learning rate update.
        V(st) = V(st) + alpha * [Gt - V(st)]
        """
        delta = alpha * (G - self.VsD[ s_hash ]) # allow key error
        self.VsD[ s_hash ] += delta
        
        self.record_changes( s_hash, delta )
        
        return abs(delta) # return the absolute value of change

    def td0_update(self, s_hash='', a_desc='', alpha=0.1, gamma=1.0, sn_hash='', reward=0.0):
        """
        Do a TD(0), Temporal-Difference-style learning rate update.
        V(st) = V(st) + alpha * [R + gamma*V(st+1) - V(st)]
        
        Note: the a_desc input is provided in order to collect transition probability data.
        """
        Vstp1 = self.VsD[ sn_hash ]
        target_val = reward + gamma*Vstp1
        delta = alpha * (target_val - self.VsD[ s_hash ]) # allow key error
        
        self.VsD[ s_hash ] += delta
        
        self.record_changes( s_hash, delta )
        
        self.save_action_results( s_hash, a_desc, sn_hash, reward)
        
        return abs(delta) # return the absolute value of change
        
    def get_Vs(self, s_hash):
        """Return the current State-Value for s_hash"""
        return self.VsD[ s_hash ] # Allow key error
        
    def set_Vs(self, s_hash, Vs):
        """Set the current State-Value for s_hash"""
        self.VsD[ s_hash ] = Vs
    
    def calc_rms_error(self, true_valueD):
        """Using the dictionary, true_valueD as reference, calc RMS error."""
        diff_sqL = []
        for s_hash, true_val in true_valueD.items():
            diff_sqL.append( (true_val - self.VsD[s_hash])**2 )
        rms = sqrt( sum( diff_sqL ) / len(diff_sqL) )
        return rms
    
    def get_biggest_action_state_err(self):
        """Estimate the biggest error in all the state values."""
        #print('self.chgTracker.get_biggest_change()', self.chgTracker.get_biggest_change())
        return self.chgTracker.get_biggest_change()

    def make_pickle_filename(self, fname):
        """Make a file name ending with .vs2_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.vs2_pickle'

        else:
            fname = fname.replace(' ','_').replace('.','_') + '.vs2_pickle'

        return fname

    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        # build name for pickle
        fname = self.make_pickle_filename( fname )

        saveD = {}
        saveD['VsD'] = self.VsD
        savedD['define_statesD'] = self.define_statesD

        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
        print('Saved StateValueColl to file:',fname)

    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle file."""

        fname = self.make_pickle_filename( fname )
        if not os.path.isfile( fname ):
            print('Pickle File NOT found:', fname)
            return False

        fileObject = open(fname,'rb')
        readD = pickle.load(fileObject)

        VsD = readD['VsD']
        define_statesD = readD['define_statesD']

        fileObject.close()
        print('Read StateValueColl from file:',fname)

        return VsD, define_statesD

    def init_from_pickle_file(self, fname=None): # pragma: no cover
        """Initialize StateValueColl from policy pickle file."""
        VsD, define_statesD = self.read_pickle_file( fname=fname )
        if VsD:
            self.VsD = VsD
            self.define_statesD = define_statesD

        self.chgTracker.clear()

    def get_policy(self):
    
        policy = Policy( environment=self.environment )
        for s_hash in self.environment.iter_all_action_states():
            a_desc = self.get_best_eps_greedy_action( s_hash, epsgreedy_obj=None )
            policy.set_sole_action( s_hash, a_desc)
        return policy
    
    def summ_print(self, fmt_V='%g', none_str='*', show_states=True,
                   show_last_change=True, show_policy=True ):
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
                        outL.append( fmt_V%self.VsD[ s_hash ] )
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
            
            if show_policy:
                policy = self.get_policy()
                policy.summ_print(verbosity=0, environment=self.environment)
        
        # ------------------------- simple output w/o a layout ------------
        else:
            lmax_hash = 6
            lmax_V = 6
            
            outL = [] # list of tuples = (s_hash, V)
            for s_hash,V in self.VsD.items():
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
    
    from introrl.policy import Policy
    from introrl.mdp_data.simple_grid_world import get_gridworld    
    
    gridworld = get_gridworld()
    policyD = gridworld.get_default_policy_desc_dict()
    
    pi = Policy( environment=gridworld )
    #pi.learn_all_states_and_actions_from_env( gridworld )
    pi.set_policy_from_piD( policyD )
    
    # -------------
    
    sv = StateValueColl( gridworld )
    
    for _ in range(10):
        sv.mc_update((0,0), 0.2, 2.0)
        sv.mc_update((0,0), 0.2, 3.0)
        sv.mc_update((0,1), 0.5, 1.0)
    print('Value at (0,0) is:', sv.get_Vs( (0,0) ) )
    print('get_biggest_action_state_err = ', sv.get_biggest_action_state_err(), '%' )
    
    sv.summ_print( fmt_V='%6g' )
    gridworld.layout = None
    sv.summ_print( fmt_V='%6g' )
