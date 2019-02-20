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

from introrl.utils.functions import select_weighted_random
from introrl.utils.gen_sort_key import NaturalOrStrKey

class StateActions( object ):
    """
    These are the actions associated with a SINGLE STATE.
    
    A Policy object will collect these into a comprehensive set of StateActions.
    
    Each State object (unless a terminal state) has an action or collection of actions
    associated with it.
    
    All actions are taken with a probability in the range 0.0 to 1.0.
    If a state has only one action, it has a probability of 1.0
    If a state has >1 actions, the sum of their probabilities is 1.0    
    """
    def __init__(self, state_obj):
        """Note that the State object typically owns this set of StateActions"""
        
        # save reference to State object.
        self.state_obj = state_obj
        
        self.action_probD = {} # index=action_obj: value=probability of action
        
        self.descD = {} # index=action_desc: value=action_obj
        
        # when changes are made to probabilty values, must normalize before
        #    StateActions is used. (i.e. all probability values must sum to 1.0)
        self.is_normalized = False

    def get_prob_weighted_action(self):
        """
        Return an Action object at random.
        Use weighted selection based on probability.
        """
        # get list of (A, prob) pairs (get_list_of_all_action_prob will normalize as reqd)
        a_probL = self.get_list_of_all_action_prob( incl_zero_prob=False )
        #print('a_probL=',a_probL)
        
        if len(a_probL) == 0:
            return None # Action object
            
        if len( a_probL ) > 1:
            (A, prob) = select_weighted_random( a_probL )
        else:
            (A, prob) = a_probL[0]

        return A
    
    def set_action_prob(self, action_obj, prob=1.0):
        """
        Will Set or Add action object and probability to action_probD. 
        
        DELAY NORMALIZING... Allows sequential adding of multipls action/prob pairs.
           i.e. Merely set the flag "is_normalized" to False 
                to trigger later "normalize" call later.
        """
        prob = float(prob)
        
        self.action_probD[ action_obj ] = float( prob ) # index=action_obj: value=probability of action
        self.descD[ action_obj.desc ] = action_obj
            
        self.is_normalized = False
    
    def set_action_prob_by_desc(self, action_desc, prob=1.0):
        """
        Will Set or Add action object and probability to action_probD. 
        
        DELAY NORMALIZING... Allows sequential adding of multipls action/prob pairs.
           i.e. Merely set the flag "is_normalized" to False 
                to trigger later "normalize" call later.
        """
        prob = float(prob)
        action_obj = self._get_action_by_desc( action_desc )
        self.action_probD[ action_obj ] = float( prob ) # index=action_obj: value=probability of action
            
        self.is_normalized = False
        
    def zero_all_actions(self):
        """Set all probabilities to 0.0  Can be useful for 1st step in initialize."""
        for A in self.action_probD.keys(): # index=action_obj: value=probability of action
            self.action_probD[A] = 0.0

    def set_sole_action(self, action_obj):
        """Set the probability of action_obj to 1.0, all others to 0.0"""
        for A in self.action_probD.keys(): # index=action_obj: value=probability of action
            self.action_probD[A] = 0.0
        self.action_probD[ action_obj ] = 1.0
        
        self.is_normalized = True
        

    def set_sole_action_by_desc(self, action_desc ):
        action_obj = self._get_action_by_desc( action_desc )
        self.set_sole_action( action_obj ) # is_normalized set in set_sole_action
        
        #print('action_obj=',action_obj)

    def initialize_sole_random(self):
        """Set a random action to be probability 1.0, all others to 0.0"""

        if len(self.action_probD) > 0:
            A = random.choice( tuple( self.action_probD.keys() ) )
            self.set_sole_action( A ) # is_normalized set in set_sole_action

    def intialize_to_equiprobable(self):
        """Make all actions be equal probability"""
        if len(self.action_probD) > 0:
            prob = 1.0 / float( len(self.action_probD) )
            for A in self.action_probD.keys(): # index=action_obj: value=probability of action
                self.action_probD[A] = prob
        
        self.is_normalized = True
            
    def has_action(self, action_obj):
        return action_obj in self.action_probD # index=action_obj: value=probability of action
    
    def get_action_prob_by_desc(self, action_desc):
        if not self.is_normalized:
            self.normalize()
        
        if action_desc in self.descD:
            A = self.descD[ action_desc ]
            prob = self.action_probD[ A ]
            return (A,prob)
        return None,None
    
    def _get_action_by_desc(self, action_desc):
        if action_desc in self.descD:
            return self.descD[ action_desc ]
        return None
        
    def has_action_desc(self, action_desc):
        action_obj = self._get_action_by_desc( action_desc )
        return self.has_action( action_obj )
    
    def remove_action(self, action_obj):
        if action_obj in self.action_probD: # index=action_obj: value=probability of action
            del self.descD[ action_obj.desc ]
            del self.action_probD[ action_obj ]
        else:
            print('WARNING... tried to remove non-existing object from StateActions')
        
        self.is_normalized = False
    
    def remove_action_by_desc(self, action_desc):
        action_obj = self._get_action_by_desc( action_desc )
        self.remove_action( action_obj ) # normalize flag set by remove_action

    def __len__(self):
        return len( self.action_probD )
    
    def get_list_of_all_action_prob(self, incl_zero_prob=False):
        """
        Return a list of all (action_obj, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        if not self.is_normalized:
            self.normalize()
        
        if incl_zero_prob:
            apL = [(A, prob) for (A, prob) in self.action_probD.items()]
        else:
            apL = [(A, prob) for (A, prob) in self.action_probD.items() if prob > 0.0]
        
        return apL
    
    def get_list_of_all_actions(self, incl_zero_prob=False):
        """
        Return a list of all action_obj objects. (do not return probability values)
        if incl_zero_prob==True, include actions with zero probability.
        """
        # don't bother to normalize, works with or w/o 
        #if not self.is_normalized:
        #    self.normalize()
        
        if incl_zero_prob:
            aL = [A for A in self.action_probD.keys()]
        else:
            aL = [A for (A, prob) in self.action_probD.items() if prob > 0.0]
        
        return aL
    
    def iter_action_prob(self, incl_zero_prob=False):
        """
        Iterate over all (action_obj, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        if not self.is_normalized:
            self.normalize()
        
        for A, prob in self.action_probD.items(): # index=action_obj: value=probability of action
            if prob > 0.0:
                yield A, prob
            elif incl_zero_prob:
                yield A, prob
    
        
    def normalize(self):
        """Scale all prob values such that they all add to 1.0"""
        self.is_normalized = True
        
        psum = 0.0
        for A, prob in self.action_probD.items():
            psum += prob
            
        if psum > 0.0:
            for A, prob in self.action_probD.items():
                self.action_probD[A] /= psum
        else:
            # if all actions are prob == 0.0, are they 0.0 on purpose?
            prob = 1.0
            if len(self.action_probD) > 0:
                prob = 1.0 / float(len(self.action_probD))
                #print('Setting all prob to:', prob)
                
            for A, p in self.action_probD.items():
                self.action_probD[A] = prob
                #print('Setting ',A.desc,' to ',prob)
            

    def summ_print(self, long=True): # pragma: no cover
        """Show actions from most to least probable."""
        print('___ %s StateActions Summary ___'%str(self.state_obj.hash) )
        print('    Nactions=%i'%len(self) )
        
        if not self.is_normalized:
            self.normalize()

        if long and len(self):
            print('         Action Probability' )
            #print('         self.action_probD =',self.action_probD)
            for (prob,desc) in sorted( [ (prob,A.desc) for (A,prob) in self.action_probD.items()], reverse=True, key=NaturalOrStrKey ):
                print('      %9s'%str(desc), '%g'%prob)
            


if __name__ == "__main__": # pragma: no cover
    
    from introrl.action import Action
    from introrl.state import State
    
    s = State( (2,2) )
    sa = StateActions( s )
    for d in ['U','D','L','R']:
        a = Action(d)
        sa.set_action_prob( a, prob=len(sa))
        
    sa.summ_print()
    
    print('-'*55)
    print( sa.get_list_of_all_action_prob() )
    print('-'*55)
    for A,prob in sa.iter_action_prob():
        print( A, prob)
    print('-'*55)
    for A,prob in sa.iter_action_prob(incl_zero_prob=True):
        print( A, prob)
    
    
    print('-'*55)
    for i in range(30):
        A = sa.get_prob_weighted_action()
        print(A.desc, end=' ')
    print()