#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

class ContinuousParameter( object ):
    """
    A Floating point parameter in a ContinuousSimulation
    """
    
    def __init__(self, name='Position', units='ft', value_init=0.0,
                 min_value=float('-inf'), max_value=float('inf')):
                     
        self.name = name
        self.units = units
        self.value = value_init
        self.value_init = value_init
        self.min_value = min_value
        self.max_value = max_value
        
    def reset(self):
        self.value = self.value_init
        
    def set_bounded_val(self, new_val):
        """Set new value, but stay in bounded region."""
        self.value = max(self.min_value, min(self.max_value, new_val) )
        
    def add_bounded_delta(self, delta):
        """Add delta to value, but stay in bounded region."""
        self.set_bounded_val( self.value + delta )
        
    def at_min_limit(self):
        return self.value <= self.min_value
        
    def at_max_limit(self):
        return self.value >= self.max_value

    def get_range_list(self, N=50):
        ansL = [ self.min_value ]
        delta = (self.max_value - self.min_value) / float(N)
        for _ in range(N):
            ansL.append( delta + ansL[-1] )
        return ansL

    def summ_print(self, pad=''):
        
        limit_str = ''
        if self.at_min_limit():
            limit_str = '(AT MIN LIMIT)'
        if self.at_max_limit():
            limit_str = '(AT MAX LIMIT)'
        
        print(pad,'--------- ContinuousParameter: "%s" ---------'%self.name)
        print(pad,' '*18,'     value:',self.value,self.units, limit_str)
        print(pad,' '*18,'value_init:',self.value_init)
        print(pad,' '*18,'     range:',self.min_value,'to',self.max_value)

if __name__=="__main__":
    
    vp = ContinuousParameter(name='Velocity', units='ft/sec', value_init=0.0,
                 min_value=-0.07, max_value=0.07)
    vp.summ_print()
    vp.add_bounded_delta(10.0)
    vp.summ_print()
    vp.add_bounded_delta(-10.0)
    vp.summ_print()
    