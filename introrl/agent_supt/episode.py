#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import sys

class Episode( object ):
    
    def __init__(self, episode_name='Generic Episode'):
        
        self.episode_name = episode_name
        
        self.is_done_flag = False
        self.episodeL = [] # list of (state, action, reward, state_next) tuples
    
    def add_delta_to_last_reward(self, delta):
        self.episodeL[-1][2] += delta
    
    def terminal_state(self):
        # only considered terminal_state if done flag was set
        if self.is_done_flag and self.episodeL:
            # get last 
            (state, action, reward, state_next) = self.episodeL[-1]
            if state_next is not None:
                return state_next # state_next of last entry
            elif state is not None:
                return state      # state of last entry where state_next is None
            else:
                return None
        else:
            return None
    
    def get_step(self, i_step):
        if i_step < len(self.episodeL):
            return self.episodeL[ i_step ]
        else:
            return (None, None, None, None)
    
    def add_step(self, state, action, reward, state_next):
        #print('Episode Added:', (state, action, reward, state_next))
        self.episodeL.append( (state, action, reward, state_next) )
    
    def __len__(self):
        return len( self.episodeL )
    
    def clear(self):
        self.is_done_flag = False
        self.episodeL = [] # list of (state, action, reward, state_next) tuples
    
    def set_done_flag(self):
        #print('Episode Ended at step #', len(self.episodeL))
        self.is_done_flag = True
    
    def get_rev_discounted_returns(self, gamma=0.9, first_visit=False, visit_type=None):
        """NOTE: 
        discounted returns (gamma*G) are calculated in reverse order of visits.
        AND, returned in REVERSED visit order.
        
        First_visit skips over states OR state-actions that have occurred previously in episode.
        
        IF first_visit==True THEN MUST INPUT visit_type = 'S' or 'SA' (for State OR State-Action)
        """
        
        # Pester the user to avoid invisible errors in getting discounted returns
        if first_visit and (visit_type not in ('S','SA')):
            raise ValueError( 'For First-Visit Discounted Returns, "visit_type" MUST be "S" or "SA"' )
        
        temp_returnsL = [] # make a temporary list of returns in case of first_visit==True
        G = 0.0
        for i, (state, action, reward, state_next) in enumerate( self.rev_iter_all_sars() ):
            G = reward + gamma * G
            temp_returnsL.append( (state, action, reward, state_next, G) )
        
        # at this point temp_returnsL is in REVERSED visit order
        
        # if first_visit==True, remove state visits after first visit.
        if first_visit:
            
            visited_set = set()
            discounted_returnsL = []
            
            # use reversed iterator to iterate the list in VISIT order (i.e. reverse the reverse)
            for (state, action, reward, state_next, G) in reversed( temp_returnsL ):
                # depending on the type of first_visit (State or State-Action) create a key
                if visit_type == 'S':
                    key = state
                else:
                    key = (state, action)
                
                # if First Visit, save visit to episodes discounted returns.
                if key not in visited_set:
                    discounted_returnsL.append( (state, action, reward, state_next, G) )
                    visited_set.add( key )
                    
            discounted_returnsL.reverse() # return in REVERSED visit order
        else:
            discounted_returnsL = temp_returnsL
            
        return discounted_returnsL

    def is_done(self):
        return self.is_done_flag
    
    def iter_first_visit_sars(self, visit_type=None):
        """
        Iterate from start to finish over episode.
        Return (state, action, reward, state_next) tuples
        ENFORCE First_visit... i.e. return ONLY 1st VISIT occurrences of (state, action, reward, state_next)
        """
        # Pester the user to avoid invisible errors in getting discounted returns
        if visit_type not in ('S','SA'):
            raise ValueError( 'For First-Visit Discounted Returns, "visit_type" MUST be "S" or "SA"' )
            
        visited_set = set()
        
        for (state, action, reward, state_next) in self.episodeL:
            if visit_type == 'S':
                key = state
            else:
                key = (state, action)
            
            if key not in visited_set:
                yield (state, action, reward, state_next)
                visited_set.add( key )
    
    def iter_all_sars(self):
        """
        Iterate from start to finish over episode.
        Return (state, action, reward, state_next) tuples
        IGNORE First_visit... i.e. return all occurrences of (state, action, reward, state_next)
        """
        for (state, action, reward, state_next) in self.episodeL:
            yield (state, action, reward, state_next)
    
    def rev_iter_all_sars(self):
        """
        Reverse-Iterate from finish to start over episode.
        Return (state, action, reward, state_next) tuples
        IGNORE First_visit... i.e. return all occurrences of (state, action, reward, state_next)
        """
        for (state, action, reward, state_next) in reversed(self.episodeL):
            yield (state, action, reward, state_next)
    
    def summ_print(self):  # pragma: no cover
        print( 'Episode:',self.episode_name, ' Length:', len( self.episodeL ) )
        for i, t in enumerate( self.episodeL ):
            print( '%3i) %s'%(i,t) )

        if self.is_done_flag:
            print('Episode Terminal, Terminal State = ', self.terminal_state())
        else:
            print('Episode Continuing.')
            
    
    
if __name__ == "__main__":  # pragma: no cover
    
    s = Episode( episode_name='Silly Episode' )
    s.add_step( 's1', 'U', 0.0, 's2' )
    s.add_step( 's2', 'R', 0.0, 's3' )
    s.add_step( 's3', 'R', 0.0, 's2' )
    s.add_step( 's2', 'R', 0.0, 's3' )
    s.add_step( 's3', 'U', 1.0, 's4' )
    s.set_done_flag()
    
    s.summ_print()
    
    print( s.episode_name, len(s), s.is_done())
    print( s.episodeL )
    print()
    print('NOT-First-Visit "S" G')
    for dr in s.get_rev_discounted_returns( gamma=0.9, first_visit=False, visit_type='S'):
        print(dr)
    print()
    print('First-Visit "S" G')
    for dr in s.get_rev_discounted_returns( gamma=0.9, first_visit=True, visit_type='S'):
        print(dr)
    print()
    print('NOT-First-Visit "SA" G')
    for dr in s.get_rev_discounted_returns( gamma=0.9, first_visit=False, visit_type='SA'):
        print(dr)
    print()
    print('First-Visit "SA" G')
    for dr in s.get_rev_discounted_returns( gamma=0.9, first_visit=True, visit_type='SA'):
        print(dr)
    
    print()
    print('FORWARD iter_first_visit_sars( visit_type="SA")')
    for tup in s.iter_first_visit_sars( visit_type="SA"):
        print( tup )
    
    