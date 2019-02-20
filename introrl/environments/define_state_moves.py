#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from introrl.reward import Reward
from introrl.utils.functions import is_float, floatCast

class DefineStateMoves( object ):
    """
    Each State s can have any number of actions and an associated 
    action probability ap of taking that action. 
    For example s can have: (a1,ap1), (a2,ap2), (a3,ap3), ... etc.

    Each combination of (State, Action) can lead to any number of next states.
    For each next state sn, there is a transition probability tp and reward R.
    Each (s,a) will have an associated triplet of (sn, tp, R).
    For example (s,a) can have: (sn1,tp1,R1), (sn2,tp2,R2), (sn3,tp3,R3), ... etc.
    
    This object helps to define the above relationships.
    """
    def __init__(self, s_hash):
        """Define the state for all actions and transitions."""
        self.s_hash = s_hash
        
        self.action_probD = {} # index=a_desc: value=action probability
        
        # snD = {} where index=snext_hash: value=(t_prob, reward_obj)
        self.action_snprD = {} # index=a_desc: value=snD (i.e. dict of possible next states.

    
    def add_action(self, a_desc, a_prob):
        """Add an action and its action probability"""
        
        # catch non-float a_prob
        if is_float( a_prob ):
            a_prob = floatCast( a_prob ) # make sure it's a simple float
        else:
            raise ValueError( 'action prob: "%s" MUST BE A FLOAT.'%str(a_prob) )
        # -----------
        
        self.action_probD[ a_desc ] = a_prob

    def add_transition(self, a_desc, snext_hash, t_prob, reward_obj):
        """Add the (sn,tp,R) triplet for the (s,a)"""

        
        # catch non-float t_prob
        if is_float( t_prob ):
            t_prob = floatCast( t_prob ) # make sure it's a simple float
        else:
            raise ValueError( 'transition prob: "%s" MUST BE A FLOAT.'%str(t_prob) )
        # -----------

        # allow float inputs for reward... recast as Reward object
        if is_float( reward_obj ):
            reward_obj = Reward( const=reward_obj )
            
        if not isinstance( reward_obj, Reward ):
            raise ValueError( 'reward_obj: "%s" MUST BE A Reward object OR float.'%str(reward_obj) )
        # -----------

        if a_desc not in self.action_snprD:
            self.action_snprD[ a_desc ] = {} # snD...  index=snext_hash: value=(t_prob, reward_obj)
            
        self.action_snprD[ a_desc ][snext_hash] = ( t_prob, reward_obj )

    def is_consistent_info(self):
        """Check to make sure all the info is consistent"""
        # check for no entries
        if (len(self.action_probD)==0) or (len(self.action_snprD)==0):
            print('ERROR... No actions or transitions given for state="%s"'%str(self.s_hash))
            return False
        
        # check for matching number of entries 
        #   --> IF ACTION IS STOCHASTIC, ONE ACTION CAN LEAD TO MANY NEXT STATES. <--
        #if len(self.action_probD) != len(self.action_snprD):
        #    print('ERROR... Num actions=%i but Num transitions=%i They MUST MATCH.'%\
        #         (len(self.action_probD), len(self.action_snprD))  )
        #    return False
        
        # check for numeric probability values and Reward objects.
        for (a_desc, a_prob) in self.action_probD.items():
            if a_desc not in self.action_probD:
                print('ERROR... Action "%s" in action probability, BUT NOT transition probability'%str(a_desc))
                return False        
        
        return True # everything looks good
        
    
    def add_to_environment(self, env):
        if not self.is_consistent_info():
            print( 'WARNING... NOT CONSISTENT. '*3 )
        
        for (a_desc, a_prob) in self.action_probD.items():
            if a_desc in self.action_snprD:
                snD = self.action_snprD[ a_desc ] # snD...  index=snext_hash: value=(t_prob, reward_obj)
                for snext_hash, (t_prob, reward_obj) in snD.items():
                    
                    #env.set_transition(self.s_hash, a_desc, 
                    #                   snext_hash, reward_obj=reward_obj, 
                    #                   action_prob=a_prob, trans_prob=t_prob)
        
                    env.TC.set_transition( self.s_hash, a_desc, 
                                           snext_hash, reward_obj=reward_obj, 
                                           action_prob=a_prob, trans_prob=t_prob)
    
        for (s_hash, a_desc, T) in env.TC.iter_all_transitions():
            T.normalize()
    
    def summ_print(self): # pragma: no cover
        print('___ "%s" State-Action, State-Transition Summary ___'%str(self.s_hash))
        if not self.is_consistent_info():
            print( 'WARNING... NOT CONSISTENT... '*2 )
        
        print('    Number of Actions = ', len(self.action_probD))
        print('    -----------------------')
        
        sum_a_prob = 0.0
        for (a_desc, a_prob) in self.action_probD.items():
            print('    Action=%s  Probability=%g'%(a_desc, a_prob))
            sum_a_prob += a_prob
            
            if a_desc in self.action_snprD:
                print('    number of transitions=%g'%len(self.action_snprD[ a_desc ])  )
                
                snD = self.action_snprD[ a_desc ] # snD...  index=snext_hash: value=(t_prob, reward_obj)
                
                sum_t_prob = 0.0
                for snext_hash, (t_prob, reward_obj) in snD.items():
                    print('        Snext=%s  Probability=%g  Reward=%s'%(str(snext_hash), t_prob, str(reward_obj)) )
                    sum_t_prob += t_prob
                if abs(sum_t_prob - 1.0) > 1.0e-4:
                    print('        NOTE: Sum of Transition Probability=%g  WILL BE NORMALIZED TO 1.0'%sum_t_prob )
                    
        if (abs(sum_a_prob - 1.0) > 1.0e-4) and (sum_a_prob > 0.0):
            print('    NOTE: Sum of Action Probability=%g  WILL BE NORMALIZED TO 1.0'%sum_a_prob )
                    
                
if __name__ == "__main__": # pragma: no cover
    
    from introrl.environments.env_baseline import EnvBaseline
            
    IO = DefineStateMoves( 'State_1' )
    #IO.summ_print()
    
    #print('_'*55)
    IO.add_action( 'U', .5)
    IO.add_action( 'D', .51)
    #IO.summ_print()
    
    
    #print('_'*55)
    IO.add_transition( 'U', (2,2), 0.2, 0.0)
    IO.add_transition( 'U', (2,0), 0.79, 1.0)
    #IO.summ_print()
    
    #print('_'*55)
    IO.add_transition( 'D', (2,0), 1.0, 1.0)
    IO.summ_print()
    
    print('_'*55)
    
    env = EnvBaseline()
    IO.add_to_environment( env )
    env.summ_print()
    
        