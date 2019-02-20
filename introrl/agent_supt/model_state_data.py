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
from introrl.utils.running_ave import RunningAve
from introrl.utils.gen_sort_key import NaturalOrStrKey

import random

class ModelStateData( object ):
    """
    For a given State, s_hash,
    this object records (Action, Reward, Next_State) data that 
    result from calling an Environment/Simulation
    
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
        
        self.action_countD = {} # index=a_desc: value=count of (s,a) occurances
        self.total_action_calls = 0 # track total number of action calls

        # sn_rD = {} where index=sn_hash: value=RunningAve of Reward (includes count)
        self.action_sn_rD = {} # index=a_desc: value=sn_rD
    
    def has_action(self, a_desc):
        return a_desc in self.action_sn_rD
    
    def get_sample_sn_r(self, a_desc):
        if a_desc in self.action_sn_rD:
            snD = self.action_sn_rD[ a_desc ] # snD...  index=sn_hash: value=rwd_ave_obj
            
            sn_hash = random.choice( tuple( snD.keys() ) )
            reward = snD[sn_hash].get_ave()
            return sn_hash, reward
        else:
            return None, None
        
    
    def get_ave_reward(self, a_desc):
        r_ave = 0.0
        if a_desc in self.action_sn_rD:
            snD = self.action_sn_rD[ a_desc ] # snD...  index=sn_hash: value=rwd_ave_obj
            a_count = self.action_countD[a_desc]
            
            for sn_hash, rwd_ave_obj in snD.items():
                # fraction of times using a_desc in s_hash resulted in sn_hash
                t_prob = float(rwd_ave_obj.num_val) / float(a_count)
                r_ave += t_prob * rwd_ave_obj.get_ave()
        return r_ave
    
    def get_action_list(self):
        return [a_desc for a_desc in self.action_countD.keys()]
        
    def get_action_desc_str(self):
        aL = sorted( [a_desc for a_desc in self.action_countD.keys()], key=NaturalOrStrKey )
        return '(' + ' '.join( [str(a) for a in aL] ) + ')'
    
    def all_state_actions_deterministic(self):
        """Check if ALL actions are Deterministic or Stochastic"""
        for snD in self.action_sn_rD.values():
            if len(snD) != 1:
                return False
            for sn_hash, R in snD.items():
                if not R.has_zero_range( max_range=1.0E-10 ):
                    return False
        return True
    
    def is_deterministic_action(self, a_desc):
        """
        Check if action is Deterministic or Stochastic
        Stochastic might mean variable Snext variable Reward or both.
        """
        snD = self.action_sn_rD[ a_desc ]
        if len(snD) != 1:
            return False
        else:
            for sn_hash, R in snD.items():
                if not R.has_zero_range( max_range=1.0E-10 ):
                    return False
        return True
    
    def get_action_stochastic_label(self, a_desc):
        if self.is_deterministic_action( a_desc ):
            return 'DETERMINISTIC'
        else:
            sL = ['STOCHASTIC'] # will add 'Snext', 'Reward' as req'd
            snD = self.action_sn_rD[ a_desc ]
            if len(snD) != 1:
                sL.append( 'SNEXT' )
            
            for sn_hash, R in snD.items():
                if not R.has_zero_range( max_range=1.0E-10 ):
                    sL.append('REWARD')
                    break
                    
            if len(sL)==3:
                return 'STOCHASTIC SNEXT & REWARD'
            return ' '.join( sL )
            
    
    def add_action(self, a_desc):
        """Add an action and initialize its action count"""
        #print('model for %s adding action %s'%(self.s_hash, a_desc))
        if a_desc not in self.action_countD:
            self.action_countD[ a_desc ] = 0 # only sets to zero 1st time.

    def save_action_results(self, a_desc, sn_hash, reward_val, 
                            force_deterministic=False):
        """
        Add sn_hash to possible next states and add to its RunningAve
        If force_deterministic is True, force the new sn_hash to be unique
        """
        
        # make sure that a_desc is initialized
        if a_desc not in self.action_countD:
            self.add_action( a_desc )
        
        # increment action counters
        self.action_countD[ a_desc ] += 1 # inc. count of a_desc calls
        self.total_action_calls += 1

        # make sure sn_hash dict is initialized for a_desc
        if a_desc not in self.action_sn_rD:
            self.action_sn_rD[ a_desc ] = {} 
            # snD... index=sn_hash: value=RunningAve of Reward
        
        # save sn_hash and update reward running average for (a_desc, sn_hash)
        if sn_hash not in self.action_sn_rD[ a_desc ]:
            self.action_sn_rD[ a_desc ][ sn_hash ] = \
                RunningAve( name= 'Reward (%s, %s, %s)'%(str(self.s_hash), str(a_desc), str(sn_hash)) )
                    
        # update the RunningAve of (a_desc, sn_hash) with current reward_val
        self.action_sn_rD[ a_desc ][sn_hash].add_val( reward_val )
        
        if force_deterministic and (len(self.action_sn_rD[ a_desc ])>1):
            # remove any sn_hash other than the current input sn_hash
            D = {sn_hash: self.action_sn_rD[ a_desc ][sn_hash]}
            self.action_sn_rD[ a_desc ] = D
            self.action_sn_rD[ a_desc ][sn_hash].set_all_attrib( 1, reward_val, reward_val, reward_val)
            

    def is_consistent_info(self):
        """Check to make sure all the info is consistent"""
        # check for no entries
        if (len(self.action_countD)==0) or (len(self.action_sn_rD)==0):
            print('ERROR... No actions or transitions given for state="%s"'%str(self.s_hash))
            return False
        
        # check for numeric probability values and Reward objects.
        for (a_desc, a_count) in self.action_countD.items():
            if a_desc not in self.action_countD:
                print('ERROR... Action "%s" in action probability, BUT NOT transition probability'%str(a_desc))
                return False        
        
        return True # everything looks good
        
    
    def add_to_environment(self, env):
        """Populate an environment object with the collected data about s_hash"""
        if not self.is_consistent_info():
            print( 'WARNING... NOT CONSISTENT. '*3 )
            
        if self.total_action_calls == 0:
            print('WARNING... No Available ModelStateData to send to Environment')
        else:
        
            for (a_desc, a_count) in self.action_countD.items():
                if a_count > 0:
                    # fraction of calls in s_hash using a_desc
                    a_prob = float(a_count) / float(self.total_action_calls)
                    
                    if a_desc in self.action_sn_rD:
                        snD = self.action_sn_rD[ a_desc ] # snD...  index=sn_hash: value=rwd_ave_obj
                        for sn_hash, rwd_ave_obj in snD.items():
                
                            # fraction of times using a_desc in s_hash resulted in sn_hash
                            t_prob = float(rwd_ave_obj.num_val) / float(a_count)
                            
                            env.TC.set_transition( self.s_hash, a_desc, 
                                                   sn_hash, reward_obj=Reward(const=rwd_ave_obj.get_ave()), 
                                                   action_prob=a_prob, trans_prob=t_prob)
            
            # make sure all normalize flags are set in env.TC
            for (s_hash, a_desc, T) in env.TC.iter_all_transitions():
                T.normalize()

    def get_state_deterministic_desc(self):
        
        if self.all_state_actions_deterministic():
            #self.action_countD = {} # index=a_desc: value=count of (s,a) occurances
            countL = []
            for a_desc, count in self.action_countD.items():
                countL.append( '%s=%i'%(a_desc, count) )
            
            return'    All DETERMINISTIC (count: %s)'%', '.join(countL)
        else:
            n_det = 0
            n_sto = 0
            for a_desc in self.action_sn_rD.keys():
                if self.is_deterministic_action( a_desc ):
                    n_det += 1
                else:
                    n_sto += 1
            if n_det == 0:
                return '    All STOCHASTIC '
            else:
                return '    (%i DETERMINISTIC and %i STOCHASTIC)'%(n_det, n_sto)
        

    def summ_print(self): # pragma: no cover
        header = 'State %s'%str(self.s_hash)
        
        #print('    ','-'*len(header))
        print('    ', header)
        if not self.is_consistent_info():
            print( 'WARNING... NOT CONSISTENT... '*2 )
        
        print('     Has %i Actions '%len(self.action_sn_rD), end=' ')
        
        print( self.get_state_deterministic_desc() )
            
        print('    ','-'*len(header))
        
        for (a_desc, a_count) in self.action_countD.items():
            a_prob = float(a_count) / float(self.total_action_calls)
            print('    Action=%s  Count=%g  Probability=%g'%(a_desc, a_count, a_prob))
            
            if a_desc in self.action_sn_rD:
                
                num_sn = len(self.action_sn_rD[ a_desc ])
                
                s = self.get_action_stochastic_label( a_desc )
                
                print('      number of next states=%g  %s'%(num_sn, s)  )
                
                snD = self.action_sn_rD[ a_desc ] # snD...  index=sn_hash: value=rwd_ave_obj
                sn_count = 0
                
                for (sn_hash, rwd_ave_obj) in snD.items():
                    sn_count += 1
                    
                    t_prob = float(rwd_ave_obj.num_val) / float(a_count)
                    
                    print('            Snext=%s  Count=%g  Prob=%g  AveRwd=%g'%\
                         (str(sn_hash), rwd_ave_obj.num_val, t_prob, rwd_ave_obj.get_ave()), end='' )
                    if rwd_ave_obj.has_zero_range( max_range=1.0E-10 ):
                        print()
                    else:
                        print('  (%g to %g) STOCHASTIC'%rwd_ave_obj.get_min_max() )
                    
                    if sn_count == len(snD):
                        print()
                    
                    
        print('____'+'_'*len(header))
                

if __name__ == "__main__": # pragma: no cover
    
    from introrl.agent_supt.model import Model
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    
    get_sim = Model( gridworld, build_initial_model=True )
    
    # ---------- make a few stochastic to test summ_print
    #get_sim.define_statesD[s_hash].save_action_results( a_desc, sn_hash, reward_val)
    
    # make just the reward stochastic
    get_sim.define_statesD[(0, 2)].save_action_results( 'R', (0,3), 2.0)
    
    # make the action stochastic
    get_sim.define_statesD[(1,0)].save_action_results( 'U', 'XXX', 0.0)
    
    # make both the action and reward stochastic
    get_sim.define_statesD[(2,2)].save_action_results( 'U', 'XXX', 2.0)
    get_sim.define_statesD[(2,2)].save_action_results( 'U', 'XXX', 2.2)

    
    
    for s_hash, rsa in get_sim.define_statesD.items():
        rsa.summ_print()


