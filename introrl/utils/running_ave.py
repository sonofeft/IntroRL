#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

class RunningAve( object ):
    """
    Keeps a running average of sampled values
    (e.g. G values from Monte Carlo episodes.)
    """
    def __init__(self, name='val' ):
        self.name = name
        self.clear()

    def clear(self):        
        self.num_val = 0
        self.ave_val = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def clone(self, new_name=''):
        if new_name:
            Rnew = RunningAve( new_name )
        else:
            Rnew = RunningAve( self.name )

        Rnew.num_val = self.num_val
        Rnew.ave_val = self.ave_val
        Rnew.min_val = self.min_val
        Rnew.max_val = self.max_val
        
        return Rnew

    def set_all_attrib(self, num_val, ave_val, min_val, max_val):
        self.num_val = num_val
        self.ave_val = ave_val
        self.min_val = min_val
        self.max_val = max_val
    
    def add_val(self, val):
        self.num_val += 1
        
        if self.num_val == 1:
            err = 0.0
            self.ave_val = val
        else:        
            err = float(val) - self.ave_val
            self.ave_val += err / float(self.num_val)

        self.min_val = min(val, self.min_val)
        self.max_val = max(val, self.max_val)
    
    def get_error_estimate(self):
        """Make a rough estimate of the error"""
        if self.num_val <= 1:
            return float('inf') # just assume huge error if no range data
        else:
            return self.get_range() / float( self.num_val )
    
    def get_est_pcent_err(self, basis=1.0): # input basis for percent calculation
        """Make a rough estimate of the percent error"""
        return 100.0 * self.get_error_estimate() / basis
    
    def get_range(self):
        return abs(self.max_val-self.min_val)
        
    def get_min_max(self):
        return self.min_val, self.max_val
    
    def has_zero_range(self, max_range=1.0E-10):
        return abs(self.max_val-self.min_val) <= max_range
    
    def get_ave(self):
        return self.ave_val
    
    def summ_print(self, basis=None): # pragma: no cover
        if basis is None:
            basis = max(self.get_range(), 1.0E-10)
        
        print('___Summary for RunningAve of "%s"___'%str(self.name))
        print('    Number of Visits =', self.num_val)
        print('    Average Value    =', self.ave_val)
        print('    Estimated Error  =', self.get_error_estimate())
        print('    Est Percent Error=', '%g%%'%self.get_est_pcent_err(basis=basis), '(basis=%g)'%basis)
        print('    Range            =', self.min_val,' to ',self.max_val)

if __name__ == "__main__": # pragma: no cover
    
    RA = RunningAve( 'TestVal' )
    for i in range( 10 ):
        RA.add_val( float(i) )
    RA.summ_print()
    
    
