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
from introrl.reward import Reward

class Transition( object ):
    """
    A Transition is defined by an intitial state S and an action A that will "transition"
    to next state Sn with reward R at probability P.
    
    In a deterministic environment, for any (S,A) pair, there is only one Sn such that the 
    probability of transition to Sn is 1.0
    
    In a stochastic environment, (S,A) can have any number of Sn.
    In that case, the sum of probabilities for all the Sn states is 1.0.
    
    Rewards can also vary in a stochastic environment.
    Reward objects can give constant, weighted tabular, or function-based float reward values.
        
    The Reward object is always associated with (s,a,sn), however, the numerical value can vary 
    with probabilty distributions of its own.
    """
    def __init__(self, state_obj, action_obj):
        """Note that the State object typically owns this set of StateActions"""
        
        self.state_obj = state_obj
        self.action_obj = action_obj
        
        self.sn_probD = {} # index=next_state_obj: value=probability of transition to Sn
        
        self.sn_hashD = {} # index=snext_hash: value=Sn (next_state_obj)
        
        # Reward objects can give constant, weighted tabular, or function-based float reward values.
        self.sn_rewardD = {} # index=next_state_obj: value=Reward object
        
        
        # when changes are made to probabilty values, must normalize before
        #    Transition is used. (i.e. all probability values must sum to 1.0)
        self.is_normalized = False
    
    def iter_sn_hash_prob_reward(self):
        """For this (s,a) transition, iterate over the next state, prob, reward"""
        if not self.is_normalized:
            self.normalize()
        
        for sn_hash, Sn in self.sn_hashD.items():
            t_prob = self.sn_probD[Sn]
            reward = self.sn_rewardD[Sn]()
            yield sn_hash, t_prob, reward
    
    
    def get_reward_obj(self, sn_hash ):
        if sn_hash in self.sn_hashD:
            next_state_obj = self.sn_hashD[ sn_hash ]
            return self.sn_rewardD[ next_state_obj ]
        else:
            return None
    
    def __len__(self):
        return len( self.sn_probD )
    
    def set_transition(self, next_state_obj, reward_obj=Reward(const=0.0), prob=1.0):
        """
        Will Set or Add prob and Reward entries to sn_probD and sn_rewardD
        
        DELAY NORMALIZING... Allows sequential adding of multipls action/prob pairs.
           i.e. Merely set the flag "is_normalized" to False 
                to trigger later "normalize" call later.
        """
        prob = float(prob)
        self.sn_probD[ next_state_obj ] = float( prob ) # index=action_obj: value=probability of action
        self.sn_hashD[ next_state_obj.hash ] = next_state_obj
        self.sn_rewardD[ next_state_obj ] = reward_obj
        self.is_normalized = False

    def set_transition_prob_by_snext_hash(self, snext_hash, prob=1.0):
        """
        Set the transition probability of the next_state_obj with snext_hash to prob.
        """
        Sn = self._get_next_state_by_hash( snext_hash )
        self.sn_probD[ Sn ] = float( prob ) # index=action_obj: value=probability of action
        
        self.is_normalized = False
                

    def set_sole_transition(self, next_state_obj):
        """Set the probability of next_state_obj to 1.0, all others to 0.0"""
        for Sn in self.sn_probD.keys(): # index=next_state_obj: value=probability of next_state_obj
            self.sn_probD[Sn] = 0.0
        self.sn_probD[ next_state_obj ] = 1.0
        
        self.is_normalized = True
    
    def _get_next_state_by_hash(self, snext_hash):
        
        if snext_hash in self.sn_hashD:
            return self.sn_hashD[ snext_hash ]
        return None

    def set_sole_transition_by_desc(self, snext_hash ):
        Sn = self._get_next_state_by_hash( snext_hash )
        self.set_sole_transition( Sn ) # is_normalized set in set_sole_transition

    def initialize_sole_random(self):
        """Set a random next state to be probability 1.0, all others to 0.0"""

        if len(self.sn_probD) > 0:
            Sn = random.choice( tuple( self.sn_probD.keys() ) )
            self.set_sole_transition( Sn ) # is_normalized set in set_sole_transition

    def intialize_to_equiprobable(self):
        """Make all next states be equal probability"""
        if len(self.sn_probD) > 0:
            prob = 1.0 / float( len(self.sn_probD) )
            for Sn in self.sn_probD.keys(): # index=next_state_obj: value=probability of next_state_obj
                self.sn_probD[Sn] = prob
        self.is_normalized = True

    def has_next_state(self, next_state_obj):
        return next_state_obj in self.sn_probD # index=next_state_obj: value=probability of next_state_obj

    def has_next_state_by_hash(self, snext_hash):
        return snext_hash in self.sn_hashD
    
    def get_next_state_prob_by_hash(self, snext_hash):
        if not self.is_normalized:
            self.normalize()
        
        if snext_hash in self.sn_hashD:
            Sn = self.sn_hashD[ snext_hash ]
            prob = self.sn_probD[ Sn ]
            return (Sn,prob)
            
        return None,None
    
    def remove_next_state(self, next_state_obj):
        if next_state_obj in self.sn_probD: # index=next_state_obj: value=probability of next_state_obj
            del self.sn_hashD[ next_state_obj.hash ]
            del self.sn_probD[ next_state_obj ]
        else:
            print('WARNING... tried to remove non-existing object from Transition')
        
        self.is_normalized = False
    
    def remove_next_state_by_hash(self, snext_hash):
        next_state_obj = self._get_next_state_by_hash( snext_hash )
        self.remove_next_state( next_state_obj ) # normalize flag set by remove_next_state

    def get_list_of_all_next_state_prob(self, incl_zero_prob=False):
        """
        Return a list of all (next_state_obj, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        if not self.is_normalized:
            self.normalize()
        
        if incl_zero_prob:
            snpL = [(Sn, prob) for (Sn, prob) in self.sn_probD.items()]
        else:
            snpL = [(Sn, prob) for (Sn, prob) in self.sn_probD.items() if prob > 0.0]
        
        return snpL
    
    def get_list_of_all_next_state(self, incl_zero_prob=False):
        """
        Return a list of all next_state_obj objects. (do not return probability values)
        if incl_zero_prob==True, include actions with zero probability.
        """
        # don't bother to normalize, works with or w/o 
        #if not self.is_normalized:
        #    self.normalize()
        
        if incl_zero_prob:
            snL = [Sn for Sn in self.sn_probD.keys()]
        else:
            snL = [Sn for (Sn, prob) in self.sn_probD.items() if prob > 0.0]
        
        return snL
    
    def iter_next_state_prob(self, incl_zero_prob=False):
        """
        Iterate over all (next_state_obj, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        if not self.is_normalized:
            self.normalize()
        
        for Sn, prob in self.sn_probD.items(): # index=next_state_obj: value=probability of next_state_obj
            if prob > 0.0:
                yield Sn, prob
            elif incl_zero_prob:
                yield Sn, prob
    

    def get_prob_weighted_next_state(self):
        """
        Return a State object at random.
        Use weighted selection based on probability.
        """
        # get list of (Sn, prob) pairs (get_list_of_all_next_state_prob will normalize as reqd)
        sn_probL = self.get_list_of_all_next_state_prob( incl_zero_prob=False )
        
        if len(sn_probL) == 0:
            return None # State object
            
        if len( sn_probL ) > 1:
            (Sn, prob) = select_weighted_random( sn_probL )
        else:
            (Sn, prob) = sn_probL[0]

        return Sn
        
    def normalize(self):
        """Scale all prob values such that they all add to 1.0"""
        self.is_normalized = True
        
        psum = 0.0
        for Sn, prob in self.sn_probD.items():
            psum += prob
            
        if psum > 0.0:
            for Sn, prob in self.sn_probD.items():
                self.sn_probD[Sn] /= psum
        else:
            # if all actions are prob == 0.0, are they 0.0 on purpose?
            prob = 1.0
            if len(self.sn_probD) > 0:
                prob = 1.0 / float(len(self.sn_probD))
                #print('Setting all prob to:', prob)
                
            for Sn, p in self.sn_probD.items():
                self.sn_probD[Sn] = prob
                #print('Setting ',Sn.hash,' to ',prob)
            

    def summ_print(self, long=True): # pragma: no cover
        """Show actions from most to least probable."""
        print('___ (%s,%s) StateAction Transition Summary ___'%(str(self.state_obj.hash), str(self.action_obj.desc)) )
        print('    Ntransitions=%i'%len(self) )
        
        if not self.is_normalized:
            self.normalize()

        if long:
            print('     Next_State Probability Reward' )
            for (prob,Sn) in sorted( [ (prob,Sn) for (Sn,prob) in self.sn_probD.items()], reverse=True ):
                R = self.sn_rewardD[Sn]
                print('      %9s'%str(Sn.hash), '%6g'%prob, '     %s'%str(R)[1:-1])
            


if __name__ == "__main__": # pragma: no cover
    from introrl.action import Action
    from introrl.state import State
    
    s = State( (2,2) )
    a = Action( 'U' )
    T = Transition(s,a)
    
    rc = Reward(const=1.1)
    reward_probL = [(0.0,1), (1.0,1), (2.0,2)]
    rt = Reward(reward_probL=reward_probL)
    
    def my_gauss():
        return random.gauss(3.0, 0.5)
    rf = Reward(reward_dist_func=my_gauss)
    
    
    T.set_transition( State( (2,3) ), reward_obj=rc, prob=0.8)
    T.set_transition( State( (1,2) ), reward_obj=rt, prob=0.1)
    T.set_transition( State( (3,2) ), reward_obj=rf, prob=0.1)
    T.set_transition( State( (0,0) ), reward_obj=rc, prob=0.0)    
    T.summ_print()
    