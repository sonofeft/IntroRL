#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object


class LearnTracker( object ):
    """
    Keep track of the number of times a (state, action) pair is used
    in a learning algorithm, like Q-learning, Sarsa, etc.
    """
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        
        # sarsn = (s_hash, a_desc, reward, sn_hash)
        self.per_episode_actionsD = {} # index=episode number, value=list of sarsn
        
        # .....  old approach inside learning routines......
        #self.steps_per_episodeL = [] # track the number of steps in each episode
        #self.reward_sum_per_episodeL = [] # track sum of rewards during each episode
        
    def add_new_episode(self):
        i = len( self.per_episode_actionsD )
        self.per_episode_actionsD[i] = [] # a list of sarsn tuples
    
    def get_episode_sarsn_list(self, i):
        return self.per_episode_actionsD[i]
    
    def record_sa(self, s_hash, a_desc):
        """Record the (State, Action)"""
        key = s_hash, a_desc
        self.saD[key] = self.saD.get(key, 0) + 1
        
    def add_sarsn_to_current_episode(self, s_hash, a_desc, reward, sn_hash):
        i = len( self.per_episode_actionsD ) - 1
        if i<0:
            self.add_new_episode()
            i = 0
        self.per_episode_actionsD[i].append( (s_hash, a_desc, reward, sn_hash) )

    def steps_per_episode(self):
        """Count the sarsn steps in per_episode_actionsD"""
        # dict hash shouldn't need to be sorted, but paranoia requires it.
        keyL = sorted( list( self.per_episode_actionsD.keys() ) )
        return [len( self.per_episode_actionsD[key] ) for key in keyL]
    
    def reward_sum_per_episode(self):
        """Add up the rewards in sarsn."""
        # dict hash shouldn't need to be sorted, but paranoia requires it.
        keyL = sorted( list( self.per_episode_actionsD.keys() ) )
        rsumL = []
        for key in keyL:
            rsum = 0.0
            for (s,a,r,sn) in self.per_episode_actionsD[key]:
                rsum += r
            rsumL.append( rsum )
        return rsumL
    
    def cum_reward_per_step(self):
        """Add up the rewards in sarsn PER STEP, NOT PER EPISODE."""
        # dict hash shouldn't need to be sorted, but paranoia requires it.
        keyL = sorted( list( self.per_episode_actionsD.keys() ) )
        cum_rsumL = []
        cum_rsum = 0.0
        for key in keyL:
            for (s,a,r,sn) in self.per_episode_actionsD[key]:
                cum_rsum += r
                cum_rsumL.append( cum_rsum )
        return cum_rsumL
    
    def iter_episodes(self):
        """iterate the episodes. an episode is a list of sasrn tuples"""
        # dict hash shouldn't need to be sorted, but paranoia requires it.
        keyL = sorted( list( self.per_episode_actionsD.keys() ) )
        for key in keyL:
            yield self.per_episode_actionsD[key]
    
    def summ_print(self):
        print('======== LearnTracker Summary =========')
        print('      Number of Episodes =', len(self.per_episode_actionsD))
        
        if len(self.per_episode_actionsD) > 0:
            min_len = len(self.per_episode_actionsD[0])
            max_len = min_len
            tot_len = 0
            
            lenL = []
            for _, L in self.per_episode_actionsD.items():
                e_len = len(L)
                min_len = min(e_len, min_len)
                max_len = max(e_len, max_len)
                tot_len += e_len
                lenL.append(e_len)
                
            print('    Min Length Episode: %6i'%min_len)
            print('    Ave Length Episode: %6.f'%( float(tot_len) / len(self.per_episode_actionsD) ) )
            print('    Max Length Episode: %6i'%max_len)
            print('    Episode Lengths',lenL)

if __name__ == "__main__":
    
    LT = LearnTracker()
    
    LT.add_new_episode()
    LT.add_sarsn_to_current_episode( 'A','Right',0.0,'A2' )
    LT.add_sarsn_to_current_episode( 'A2','Right',0.0,'A3' )
    LT.add_sarsn_to_current_episode( 'A3','Up',1.0,'X' )

    LT.add_new_episode()
    LT.add_sarsn_to_current_episode( 'B','Right',0.0,'B2' )
    LT.add_sarsn_to_current_episode( 'B2','Right',0.0,'B3' )
    LT.add_sarsn_to_current_episode( 'B3','Stay',0.0,'B3' )
    LT.add_sarsn_to_current_episode( 'B3','Up',1.0,'X' )

    LT.summ_print()    
