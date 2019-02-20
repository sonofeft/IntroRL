#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import math

class RunningStats:

    def __init__(self, name='val'):
        self.name = name
        self.clear()

    def clear(self):
        self.num_val = 0
        self.old_m = 0.0
        self.new_m = 0.0
        self.old_s = 0.0
        self.new_s = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def add_val(self, x):
        self.num_val += 1

        self.min_val = min(x, self.min_val)
        self.max_val = max(x, self.max_val)

        if self.num_val == 1:
            self.old_m = self.new_m = x
            self.old_s = 0.0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.num_val
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s
    
    def get_error_estimate(self):
        """Make a rough estimate of the error"""
        if self.num_val <= 1:
            return float('inf') # just assume huge error if no range data
        else:
            return self.std_dev()
    
    def get_est_pcent_err(self, basis=1.0): # input basis for percent calculation
        """Make a rough estimate of the percent error"""
        return 100.0 * self.get_error_estimate() / basis
    
    def get_range(self):
        return abs(self.max_val-self.min_val)
        
    def get_min_max(self):
        return self.min_val, self.max_val
    
    def has_zero_range(self, max_range=1.0E-10):
        return abs(self.max_val-self.min_val) <= max_range
    

    def get_ave(self): # redundant call to mean.
        return self.mean()

    def mean(self):
        return self.new_m if self.num_val else 0.0

    def variance(self):
        return self.new_s / (self.num_val - 1) if self.num_val > 1 else 0.0

    def std_dev(self):
        return math.sqrt(self.variance())
    
    def summ_print(self, basis=None): # pragma: no cover
        if basis is None:
            basis = max(self.get_range(), 1.0E-10)

        print('___Summary for RunningStats of "%s"___'%self.name)
        print('    Number of Visits =', self.num_val)
        print('    Mean Value       =', self.mean())
        print('    Estimated Error  =', self.get_error_estimate())
        print('    Est Percent Error=', '%g%%'%self.get_est_pcent_err(basis=basis), '(basis=%g)'%basis)
        print('    Variance         =', self.variance())
        print('    Standard Dev.    =', self.std_dev())
        print('    Range            =', self.min_val,' to ',self.max_val)

if __name__ == "__main__": # pragma: no cover
    
    import random
    
    RA = RunningStats( 'TestVal' )
    for i in range( 1000 ):
        RA.add_val( random.gauss(3.0, 0.5) )
    RA.summ_print()
    print('-----------------------------------------')
    RA.clear()
    for i in range( 10 ):
        RA.add_val( float(i) )
    RA.summ_print()
    
    
    
