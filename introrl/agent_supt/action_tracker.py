#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object


class ActionTracker( object ):
    """
    Keep track of the number of times a (state, action) pair is used
    in a learning algorithm, like Q-learning, Sarsa, etc.
    """
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.saD = {} # index = (s,a), value=count
    
    def get_count(self, s_hash, a_desc):
        """Return the count of the number of times (s,a) was used."""
        return self.saD.get( (s_hash, a_desc), 0)
    
    def record_sa(self, s_hash, a_desc):
        """Record the (State, Action)"""
        key = s_hash, a_desc
        self.saD[key] = self.saD.get(key, 0) + 1
                
    def summ_print(self):
        print(' (state, action)    COUNT')
        for key, count in self.saD.items():
            print('%18s'%str(key),' ',count)

if __name__ == "__main__":
    
    AT = ActionTracker()
    
    AT.record_sa('A','Left')
    AT.record_sa('A','Left')
    AT.record_sa('A','Right')
    AT.record_sa('A','Left')
    AT.record_sa('A','Left')

    AT.record_sa('B',1)
    AT.record_sa('B',4)
    AT.record_sa('B',2)
    AT.record_sa('B',2)


    AT.summ_print()    
