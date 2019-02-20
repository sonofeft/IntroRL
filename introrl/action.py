#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

class Action( object ):
    """
    A Action has the following properties:
    description = immutable representation, typically a string description
    """
        
    def __init__(self, description):
        self.desc = description # immutable representation
        
    def __str__(self):
        return '<Action "%s">'%( str(self.desc) )


if __name__ == "__main__": # pragma: no cover
    
    a = Action( 'right' )
    print( a )