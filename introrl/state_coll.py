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
from introrl.state import State
from introrl.utils.gen_sort_key import NaturalOrStrKey

class StateColl( object ):
    """
    An StateColl holds all states in Environment.
    """
    
    def __init__(self, name='States'):
        
        self.stateD = {} # index=state hash, value=State object
        
        self.co_indexD = {} # index=state hash, value=creation order integer index
    
    def _make_State_obj(self, state_hash):
        # set index to current length of self.co_indexD (creation order integer index)
        self.co_indexD[ state_hash ] = len( self.stateD )
        
        S = State( state_hash )
        self.stateD[ state_hash ] = S
        return S
    
    def get_state_co_index(self, state_hash):
        if state_hash in self.co_indexD: # creation order integer index
            return self.co_indexD[ state_hash ]
        else:
            return None
    
    def add_state(self, state_hash):
        return self.get_state_obj( state_hash )
        
    def has_state_hash(self, state_hash):
        return state_hash in self.stateD
    
    def __len__(self):
        return len( self.stateD )
    
    def get_random_state(self):
        if len(self.stateD) > 0:
            return random.choice( tuple( self.stateD.values() ) )
        else:
            return None
    
    def iter_states(self):
        for S in self.stateD.values():
            yield  S
    
    def iter_state_hash(self):
        for S in self.stateD.values():
            yield  S.hash
    
    def iter_sorted_state_hash(self):
        for (hash,S) in sorted( self.stateD.items(), key=NaturalOrStrKey ):
            yield  hash
    
    def get_full_shash_list(self):
        return sorted( [S.hash for S in self.iter_states()], key=NaturalOrStrKey )
        #return sorted( [str(S.hash) for S in self.iter_states()] )
    
    def get_state_obj(self, state_hash):
        if state_hash not in self.stateD:
            self._make_State_obj( state_hash )
            
        return self.stateD[ state_hash ]

    def summ_print(self, long=True, terminal_set=None): # pragma: no cover
        """
        If terminal_set is provided (a set of State objects),
        mark those states as terminal states.
        """
        print('___ StateColl Summary ___')
        print('    Nstates=%i'%len(self.stateD) )

        if long:
            max_hash = 4
            for S in self.stateD.values():
                max_hash = max(max_hash, len(str(S.hash)) )
            fmt_hash = '%' + '%is'%max_hash
            
            print('  state_hash' )
            #for state_hash in sorted( [ str(hash) for hash in self.stateD.keys() ] ):
            for state_hash in sorted( [ hash for hash in self.stateD.keys() ], key=NaturalOrStrKey ):
                S = self.get_state_obj( state_hash )
                termess = ''
                if terminal_set is not None:
                    if S.hash in terminal_set:
                        termess = ' TERMINAL'
            
                print('    ', fmt_hash%str(S.hash), termess   )


if __name__ == "__main__": # pragma: no cover
    
        
    sc = StateColl( name='Testing' )
    for d in ['----X---O', ('a',3), 47]:
        print( sc.get_state_obj( d ) )
    
    sc.summ_print()
    print('-'*55)
    print()
