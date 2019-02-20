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

class State( object ):
    """
    A State has the following properties:
    state_hash  = immutable representation of State (used to rebuild full State Model)
    """
            
    def __init__(self, state_hash):
        self.hash = state_hash  # immutable representation
        
    def __str__(self):
        return '<State "%s">'%( str(self.hash) )

        
    def summ_print(self): # pragma: no cover
        
        print( str(self) )
        

if __name__ == "__main__": # pragma: no cover
    
    s = State( 'X-OX-OX--' )
    print( s )    
    s = State( (1,'Z',5.5) )
    print( s )