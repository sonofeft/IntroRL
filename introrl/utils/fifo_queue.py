#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object



class FiFoActionQueue:
    """
    A FiFo Queue that holds actions taken during an episode.
    Intended to hold each agents most recent move.
    Maxsize operates differently than normal Queue.  Here it pops next item until == maxsize.
    """
    def __init__(self, maxsize=0):
        self.itemL = []
        self.maxsize = maxsize

    def clear(self):
        self.itemL = []

    def isEmpty(self):
        return self.itemL == []

    def put(self, item):
        self.itemL.insert(0,item)
        
        if self.maxsize: # if 0 then ignore
            while len(self.itemL) > self.maxsize:
                self.itemL.pop() # Throw away next item on the FiFoActionQueue

    def pop(self):
        return self.itemL.pop()
        
    def pop_ordered_list(self):
        """Without changing Queue, return a list of items in pop order."""
        return list(reversed( self.itemL ))
        
    def pop_ordered_iter(self):
        """Without changing Queue, return a list of items in pop order."""
        return reversed( self.itemL )

    def qsize(self):
        return len(self.itemL)
    
    def __len__(self):
        return len(self.itemL)
    
    def __getitem__(self, i):
        """index 0 is next item to be popped off FiFoActionQueue."""
        if i<0:
            j = -i -1
        else:
            j = len(self.itemL) - i - 1
        return self.itemL[j]
    
    def empty(self):
        return len(self.itemL) == 0

if __name__ == "__main__": # pragma: no cover

    print('--------- FiFoActionQueue -----------')

    q = FiFoActionQueue( maxsize=3 )

    for i in range(5):
        q.put(i)

    for i in range(q.qsize()):
        print( q[i], end='' )
    print(' Index Order')
    print( q[-1], ' Last Index Item' )

    print( q.pop_ordered_list(), ' Pop Ordered List' )
    for v in q.pop_ordered_iter():
        print(v, end='')
    print(' Pop Ordered Iterator')

    while not q.empty():
        print( q.pop(), end='' )
    print(' Pop Order')
    print( q.pop_ordered_list(), ' Pop Ordered List' )
