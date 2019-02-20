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
from introrl.action import Action
from introrl.utils.gen_sort_key import NaturalOrStrKey

class ActionColl( object ):
    """
    An ActionColl holds all possible actions in Environment.
    """
    
    def __init__(self, name='Actions'):
        
        self.actionD = {} # index=action description, value=Action object
        
    def _make_Action_obj(self, description):
        A = Action( description )
        self.actionD[ description ] = A
        return A
        
    def add_action(self, description):
        return self.get_action_obj( description )
        
    def has_action_desc(self, description):
        return description in self.actionD
    
    def get_random_action(self):
        return random.choice( tuple( self.actionD.values() ) )
    
    def iter_actions(self):
        for A in self.actionD.values():
            yield A
        
    def __len__(self):
        return len( self.actionD )
            
    def get_action_obj(self, description):
        if description not in self.actionD:
            self._make_Action_obj( description )
        return self.actionD[ description ]

    def summ_print(self, long=True): # pragma: no cover
        print('___ ActionColl Summary ___')
        print('    Nactions=%i'%len(self.actionD) )

        if long:
            print('    Description' )
            for desc in sorted( [desc for desc in self.actionD.keys()], key=NaturalOrStrKey ):
                print('   %9s'%str(desc))
            

if __name__ == "__main__": # pragma: no cover
    
    ac = ActionColl( name='Testing' )
    for d in ['U','D','L','R']:
        print( ac.get_action_obj( d ) )
    
    ac.summ_print()
    print('-'*25,'iter_actions')
    for A in ac.iter_actions():
        print(A, end=' ')
    print()
        
    