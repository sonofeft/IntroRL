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

from introrl.action import Action
from introrl.action_coll import ActionColl

from introrl.state import State
from introrl.state_coll import StateColl

from introrl.state_actions import StateActions
from introrl.utils.gen_sort_key import NaturalOrStrKey

class StateActionsColl( object ):
    """
    A StateActionsColl holds all State-Actions in Environment.
    
    Each State object (unless a terminal state) has an action or collection of actions
    associated with it.
    
    All actions are taken with a probability in the range 0.0 to 1.0.
    If a state has only one action, it has a probability of 1.0
    If a state has >1 actions, the sum of their probabilities is 1.0    
    """
    
    def __init__(self, name='State-Actions'):
        
        self.name = name
        
        self.state_actionsD = {} # index=state object, value=StateActions object
        
        self.action_coll = ActionColl( name=name + ' Actions' )
        self.state_coll  =  StateColl( name=name + ' States' )

    def has_state_action(self, state_hash):
        """Check to see if state-action is defined."""
        if self.state_coll.has_state_hash( state_hash ):
            S = self.state_coll.get_state_obj( state_hash )
            if S in self.state_actionsD:
                return True
        return False

    def get_SA_object(self, state_hash):
        if not self.has_state_action( state_hash ):
            return None
        
        SA = self.get_state_action_obj( state_hash )
        return SA

    def get_prob_weighted_action_desc(self, state_hash):
        A = self.get_prob_weighted_action(  state_hash )
        if A is not None:
            return A.desc
        else:
            return None

    def __getitem__(self, state_hash):
        return self.get_prob_weighted_action_desc( state_hash )
        
    def get(self, state_hash, none_rtn=None):
        """A dictionary-like get function for finding action descriptions"""
        a_desc = self.get_prob_weighted_action_desc( state_hash )
        if a_desc is None:
            a_desc = none_rtn
        return a_desc
        

    def get_prob_weighted_action(self, state_hash):
        """
        Return an Action object at random.
        Use weighted selection based on probability.
        """
        #print( '  SAC.state_coll.has_state_hash( "%s" )'%str(state_hash),self.state_coll.has_state_hash( state_hash ) )
        if not self.has_state_action( state_hash ):
            return None
        
        SA = self.get_state_action_obj( state_hash )
        return SA.get_prob_weighted_action()

    def set_action_prob(self, state_hash, action_desc, prob=1.0):
        """
        Will Set or Add action object and probability within StateActions object. 
        """
        A = self.action_coll.get_action_obj( action_desc )
        SA = self.get_state_action_obj( state_hash )
        
        SA.set_action_prob( A, prob=prob)

    def set_sole_action(self, state_hash, action_desc):
        """Set the probability of action_obj to 1.0, all others to 0.0"""
        
        SA = self.get_state_action_obj( state_hash )
        SA.set_sole_action_by_desc( action_desc )

    def intialize_to_all_zero_prob(self):
        """Set all action probabilities throughout the model to 0.0"""
        for S, SA in self.state_actionsD.items():
            SA.zero_all_actions()

    def initialize_sole_random(self, state_hash):
        """Set a random action to be probability 1.0, all others to 0.0"""

        SA = self.get_state_action_obj( state_hash )
        SA.initialize_sole_random()

    def intialize_to_equiprobable(self, state_hash):
        """Make all actions be equal probability"""
        SA = self.get_state_action_obj( state_hash )
        SA.intialize_to_equiprobable()

    def has_action(self, state_hash, action_desc):
        """Check if state_hash has action_desc."""
        if not self.has_state_action( state_hash ):
            return False
        if not self.action_coll.has_action_desc( action_desc ):
            return False
        
        A = self.action_coll.get_action_obj( action_desc )
        SA = self.get_state_action_obj( state_hash )
        return SA.has_action( A )

    def get_action_prob(self, state_hash, action_desc):
        """Return the probability of action from state"""
        if not self.has_state_action( state_hash ):
            return None
        if not self.action_coll.has_action_desc( action_desc ):
            return None
            
        SA = self.get_state_action_obj( state_hash )
        A,prob = SA.get_action_prob_by_desc( action_desc ) # normalizes as reqd
        return prob

    def remove_action(self, state_hash, action_desc):
        """If state has action, remove it."""
        SA = self.get_state_action_obj( state_hash )
        SA.remove_action_by_desc( action_desc )

    def get_list_of_all_action_desc_prob(self, state_hash, incl_zero_prob=False):
        """
        Return a list of all (action_desc, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        SA = self.get_state_action_obj( state_hash )
        
        # get list of (action_obj, prob) pairs
        apL = SA.get_list_of_all_action_prob( incl_zero_prob=incl_zero_prob)
        return [ (A.desc, prob) for (A,prob) in apL ]

    def get_list_of_all_action_desc(self, state_hash, incl_zero_prob=False):
        """
        Return a list of all action descriptions. (do not return probability values)
        if incl_zero_prob==True, include actions with zero probability.
        """
        SA = self.get_state_action_obj( state_hash )
        
        # get list of Action objects
        aL = SA.get_list_of_all_actions( incl_zero_prob=incl_zero_prob )
        return [ A.desc for A in aL ]

    def iter_action_desc_prob(self, state_hash, incl_zero_prob=False):
        """
        Iterate over all (action_desc, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        SA = self.get_state_action_obj( state_hash )
        
        for (A, prob) in SA.iter_action_prob( incl_zero_prob=incl_zero_prob ):
            if A is None:
                yield None, None
            else:
                yield A.desc, prob

    def iter_prob_sorted_adesc_prob(self, state_hash, incl_zero_prob=False):
        """
        Iterate over all (action_desc, prob) pairs. SORTED BY PROBABILITY (high to low)
        if incl_zero_prob==True, include actions with zero probability.
        """
        SA = self.get_state_action_obj( state_hash )
        paL = sorted( [(prob,A) for (A,prob) in SA.iter_action_prob( incl_zero_prob=incl_zero_prob ) if A is not None],\
                      reverse=True, key=NaturalOrStrKey)
        
        for (prob, A) in paL:
            yield A.desc, prob

    def iter_shash_adesc_prob(self, incl_zero_prob=False):
        #         self.state_actionsD = {} # index=state object, value=StateActions object
        for S, SA in self.state_actionsD.items():
            for (adesc, prob) in self.iter_action_desc_prob( S.hash, incl_zero_prob=incl_zero_prob):
                yield S.hash, adesc, prob
        
    def get_state_action_obj(self, state_hash):
        """Return the StateActions object (will create if necessary)"""
        #print( '1)',self.state_coll.get_full_shash_list(), id(self.state_coll) )
        S = self.state_coll.get_state_obj( state_hash )
        #print( '    2)',self.state_coll.get_full_shash_list() )
        
        if S not in self.state_actionsD:
            SA = StateActions( S )
            self.state_actionsD[ S ] = SA
        else:
            SA = self.state_actionsD[ S ]
        
        return SA
        
    def add_state_action(self, state_hash):
        return self.get_state_action_obj( state_hash )
    
    def __len__(self):
        return len( self.state_actionsD )

    def get_state_summ_str(self, s_hash, verbosity=1):
        
        # get list of (a_desc, prob) tuples
        apL = self.get_list_of_all_action_desc_prob( s_hash, incl_zero_prob=False)
        if len(apL)==1:
            return str( apL[0][0] )
        
        # reverse the tuple order so it can be sorted by descending prob
        paL = sorted( [(prob, a_desc) for (a_desc, prob) in apL ], reverse=True, key=NaturalOrStrKey )
        
        if verbosity >= 1:
            return ''.join( ['%s(%g)'%(astr, prob) for (prob, astr) in paL] )
        else:
            return '(' + ' '.join( ['%s'%(astr, ) for (prob, astr) in paL] ) + ')'
            
        

    def summ_print(self, long=True): # pragma: no cover
        """Show state-actions in sorted order."""
        print('=== %s StateActionsColl Summary ==='%str(self.name) )
        print('    Nstate-actions=%i\n'%len( self ) )

        if long:
            s_hashL = sorted( [(S.hash,S) for S in self.state_actionsD.keys()], key=NaturalOrStrKey )
            
            for (s_hash,S) in s_hashL:
                SA = self.state_actionsD[ S ]
                SA.summ_print()
            

if __name__ == "__main__": # pragma: no cover
    
    sca = StateActionsColl()
    
    for s_hash in ['----X---O', ('a',3), 47]:
        sca.add_state_action( s_hash )

        for a_desc in ['U','D','L','R']:
            sca.set_action_prob( s_hash, a_desc, prob=1.0)
            
    sca.summ_print()
    