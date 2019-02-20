#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

"""
Created on Tue Jan 15 21:08:15 2019

@author: charlie
"""

from introrl.utils.sorteddict import SortedDict

class ChangeTracker( object ):
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.SD = SortedDict() # index=value, value=count (number of times submitted)
        
    def get_biggest_change(self):
        """Return the biggest change recorded."""
        if self.SD:
            return self.SD.peekitem(-1)[0]
        else:
            return float('inf')
    
    def get_average_change(self):
        """Return the average of all the changes"""
        if self.SD:
            Ntot = sum( [N for val,N in self.SD.items()] )
            NxV  = sum( [N*val for val,N in self.SD.items()] )
            return float(NxV) / float(Ntot)
        else:
            return float('inf')
    
    def get_number_of_changes(self):
        """Return the total number of changes recorded"""
        if self.SD:
            Ntot = sum( [N for val,N in self.SD.items()] )
            return Ntot
        else:
            return 0
    
    def inc_change(self, value):
        """If value is in SD, increment the count, otherwise add it"""
        if value in self.SD:
            self.SD[value] += 1
        else:
            self.SD[value] = 1
            
    def dec_change(self, value):
        """If value is in SD, decrement the count.  If results in 0, remove it."""
        if value in self.SD:
            self.SD[value] -= 1
            if self.SD[value] <= 0:
                del self.SD[value]
                
    def summ_print(self):
        print('  KEY    COUNT')
        for key, count in self.SD.items():
            print(key,' ',count)

if __name__ == "__main__":
    
    CT = ChangeTracker()
    
    CT.inc_change(1.1)    
    CT.inc_change(1.1)    
    CT.inc_change(1.1)    
    CT.inc_change(1.1)    
    CT.inc_change(1.1)    
    CT.inc_change(1.1)    
    CT.inc_change(1.011)    
    CT.inc_change(1.1)    
    CT.inc_change(1.21)    

    CT.summ_print()    
    print('get_biggest_change=', CT.get_biggest_change())
    print('-'*55)
    CT.dec_change(1.1)
    CT.summ_print()    
    print('get_average_change=', CT.get_average_change())
    print('get_number_of_changes=', CT.get_number_of_changes())
    CT.clear()    
    CT.summ_print()    
    print('-'*55)
    
    print('get_biggest_change=', CT.get_biggest_change())
    print('get_average_change=', CT.get_average_change())
    print('get_number_of_changes=', CT.get_number_of_changes())
    