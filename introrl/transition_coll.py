#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from introrl.state_actions_coll import StateActionsColl
from introrl.transition import Transition
from introrl.reward import Reward

class TransitionColl( object ):
    """
    A TransitionColl object defines all the possible state-to-state 
    transitions in an Environment
    
    A transition is defined by an intitial state S, an action A that will "transition"
    to next state Sn with reward R at probability P.
    
    In a deterministic environment, for any (S,A) pair, there is only one Sn such that the 
    probability of transition to Sn is 1.0
    
    In a stochastic environment, (S,A) can have any number of Sn.
    In that case, the sum of probabilities for all the Sn states is 1.0.
    """
    
    def __init__(self, name='Generic Environment'):
        
        self.name = name
        
        self.transitionsD = {} # index=(s_hash, a_desc): value=Transition object
        
        self.sa_coll = StateActionsColl( name=name + ' State-Actions' )
        
        # get action and state collections from the StateActionsColl
        self.action_coll = self.sa_coll.action_coll # share action_coll with StateActionsColl
        self.state_coll  =  self.sa_coll.state_coll # share state_coll with StateActionsColl
    
    def iter_all_transitions(self):
        for (s_hash, a_desc), T in self.transitionsD.items():
            yield s_hash, a_desc, T

        
    def get_terminal_set_and_action_set(self):
        """Either simple terminal or infinite loop to itself"""
        terminal_set = set()
        action_set = set()
        
        for S in self.state_coll.iter_states():
            is_term = True
            action_set_v1 = set( self.sa_coll.get_list_of_all_action_desc( S.hash, incl_zero_prob=True) )
            #print('Step 1 action_set_v1 =', action_set_v1)
            
            for a_desc in action_set_v1:
                for (Sn,prob) in self.iter_next_state_prob( S.hash, a_desc, incl_zero_prob=True):
                    if Sn.hash != S.hash:
                        is_term = False
            if is_term:
                terminal_set.add( S.hash )
                #print( '    type(S.hash) =',type(S.hash),'  S.hash=',S.hash )
            else:
                action_set.add( S.hash )
                    
        #print('On Exit terminal_set =', terminal_set)
        return terminal_set, action_set
        
    
    def set_transition(self, s_hash, a_desc, 
                       snext_hash, reward_obj=Reward(const=0.0), 
                       action_prob=1.0, trans_prob=1.0):
        """
        Will Set or Add prob and Reward entries to sn_probD and sn_rewardD
        
        action_prob controls the probability of picking an action from a list of actions.
        i.e. if in state s, there can be a list of (a1,p1), (a2,p2), (a3,p3), etc.
        
        trans_prob controls the probability of picking next state from a list of next states.
        i.e. if taking action a in state s, there can be a list of (sn1,p1), (sn2,p2), (sn3,p3), etc.
    
        Rewards can vary in a stochastic environment.
        Reward objects can give constant, weighted tabular, or function-based float reward values.
        
        The Reward object is always associated with (s,a,sn), however, the numerical value can vary 
        with probabilty distributions of its own.
        
        DELAY NORMALIZING... Allows sequential adding of multipls action/prob pairs.
           i.e. Merely set the flag "is_normalized" to False 
                to trigger later "normalize" call.
        """
        self.sa_coll.set_action_prob( s_hash, a_desc, prob=action_prob)
        
        T = self.get_transition_obj( s_hash, a_desc )
        Sn = self.state_coll.get_state_obj( snext_hash )
        
        T.set_transition( Sn, reward_obj=reward_obj, prob=trans_prob)
    
    def _make_Transition_obj(self, s_hash, a_desc ):
        S  = self.state_coll.get_state_obj( s_hash )
        A  = self.action_coll.get_action_obj( a_desc )
        T = Transition( S, A )
        self.transitionsD[ (s_hash, a_desc) ] = T
        return S
        
    def get_transition_obj(self, s_hash, a_desc):
        if (s_hash, a_desc) not in self.transitionsD:
            self._make_Transition_obj( s_hash, a_desc )
        return self.transitionsD[ (s_hash, a_desc) ]

    
    def set_transition_prob(self, s_hash, a_desc, snext_hash, prob=1.0):
        """
        Set the transition probability of (s_hash, a_desc, snext_hash)  to prob.
        """        
        T = self.transitionsD[ (s_hash, a_desc) ]
        T.set_transition_prob_by_snext_hash( snext_hash, prob=prob)

    def set_sole_transition(self,  s_hash, a_desc, snext_hash):
        """
        Set the probability of (s_hash, a_desc, snext_hash) to 1.0, 
            all other (s_hash, a_desc, snext_hash) to 0.0
        """
        Sn = self.state_coll.get_state_obj( snext_hash )
        T = self.transitionsD[ (s_hash, a_desc) ]
        T.set_sole_transition( Sn )
        
    def initialize_sole_random(self,  s_hash, a_desc):
        """
        Set a random (s_hash, a_desc) state to be probability 1.0, 
           all other (s_hash, a_desc) to 0.0
        """
        T = self.transitionsD[ (s_hash, a_desc) ]
        T.initialize_sole_random()

    def intialize_to_equiprobable(self,  s_hash, a_desc):
        """Make all (s_hash, a_desc) states to be equal probability"""
        T = self.transitionsD[ (s_hash, a_desc) ]
        T.intialize_to_equiprobable()

    def has_next_state(self, s_hash, a_desc, snext_hash):
        T = self.transitionsD[ (s_hash, a_desc) ]
        Sn = self.state_coll.get_state_obj( snext_hash )
        return T.has_next_state( Sn )

    def get_next_state_prob(self, s_hash, a_desc, snext_hash):
        T = self.transitionsD[ (s_hash, a_desc) ]
        return T.get_next_state_prob_by_hash( snext_hash )

    def remove_next_state(self, s_hash, a_desc, snext_hash):
        T = self.transitionsD[ (s_hash, a_desc) ]
        Sn = self.state_coll.get_state_obj( snext_hash )
        T.remove_next_state( Sn )

    def get_list_of_all_next_state_prob(self, s_hash, a_desc, incl_zero_prob=False):
        """
        Return a list of all (next_state_obj, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        T = self.transitionsD[ (s_hash, a_desc) ]
        return T.get_list_of_all_next_state_prob( incl_zero_prob=incl_zero_prob )
        
    def get_list_of_all_next_state(self, s_hash, a_desc, incl_zero_prob=False):
        """
        Return a list of all next_state_obj objects. (do not return probability values)
        if incl_zero_prob==True, include actions with zero probability.
        """
        T = self.transitionsD[ (s_hash, a_desc) ]
        return T.get_list_of_all_next_state( incl_zero_prob=incl_zero_prob )

    def iter_next_state_prob(self, s_hash, a_desc, incl_zero_prob=False):
        """
        Iterate over all (next_state_obj, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        T = self.transitionsD[ (s_hash, a_desc) ]
        for (Sn, prob) in T.iter_next_state_prob(  incl_zero_prob=incl_zero_prob  ):
            yield Sn, prob

    def get_prob_weighted_next_state_hash(self, s_hash, a_desc):
        """Return the next state hash """
        Sn = self.get_prob_weighted_next_state( s_hash, a_desc )
        if Sn is not None:
            return Sn.hash
        else:
            return None

    def get_prob_weighted_next_state(self, s_hash, a_desc):
        """
        Return a State object at random.
        Use weighted selection based on probability.
        """
        if (s_hash, a_desc) in self.transitionsD:
            T = self.transitionsD[ (s_hash, a_desc) ]
            return T.get_prob_weighted_next_state()
        else:
            return None

    def get_reward_value(self, s_hash, a_desc, sn_hash):
        """Return the float value of Reward """
        if (s_hash, a_desc) in self.transitionsD:
            T = self.transitionsD[ (s_hash, a_desc) ]
            Robj = T.get_reward_obj( sn_hash )
            if Robj is None:
                return None
            else:
                return Robj() # call the Reward object.
        else:
            return None

    def __len__(self):
        return len( self.transitionsD )

    def summ_print(self, long=True): # pragma: no cover
        """Show state-actions in sorted order."""
        print('=== %s TransitionColl Summary ==='%str(self.name) )
        print('    Nstate-actions=%i'%len( self.sa_coll ) )
        print('    Ntransitions=%i\n'%len( self ) )

        if long:
            #self.sa_coll.summ_print()
            term_set, action_set = self.get_terminal_set_and_action_set()
            
            for s_hash in self.state_coll.iter_sorted_state_hash():
                if self.sa_coll.has_state_action( s_hash ):
                    print('___ %s StateActions Summary ___'%str(s_hash) )
                    SA = self.sa_coll.get_SA_object( s_hash )
                    
                    if s_hash in term_set:
                        print('    Nactions=%i --> TERMINAL'%len(SA) )
                    else:
                        print('    Nactions=%i'%len(SA) )
                    
                    if len(SA):
                        print('         Action Probability                 (Next, Reward, prob)' )
                        for (a_desc, a_prob) in self.sa_coll.iter_prob_sorted_adesc_prob( s_hash, incl_zero_prob=True):
                        
                            transitionL = [] # holds transition description strings.
                            if self.sa_coll.has_action( s_hash, a_desc):
                                T = self.transitionsD[ (s_hash, a_desc) ]
                                for (sn_hash, t_prob, reward) in T.iter_sn_hash_prob_reward():
                                    transitionL.append( '(%s %g_R %g)'%(str(sn_hash), reward, t_prob) )
                            
                            print('      %9s'%a_desc, '%9.6f'%a_prob, '  Next State(s) =', ' | '.join(transitionL))




if __name__ == "__main__": # pragma: no cover
    
    TC = TransitionColl( name='Testing' )
            
    actionD = {'B':  (1, 4),
               'C': (1, 5),
               'D':  (1, 7)  }
                   
    rewardD = {'A': -1.0}
        
    def add_event( s_hash, a_desc, sn_hash ):
        
        r = rewardD.get( sn_hash, 0.0)
            
        TC.set_transition( s_hash, a_desc,
                           sn_hash, reward_obj=Reward(const=r), 
                           action_prob=1.0, trans_prob=1.0)
    
    add_event( 'B', 4, 'A' )
    add_event( 'B', 1, 'C' )
    add_event( 'B', 1, 'D' ) # same action, different s_next
    
    add_event( 'C', 5, 'B' )
    add_event( 'C', 1, 'D' )

    
    TC.summ_print()
    