#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from introrl.utils.functions import select_weighted_random

CONST = 1
TABLE = 2
FUNC = 3

class Reward( object ):
    """
    A transition from one state to another returns a reward value.
    A (State, Action) pair (S,A), initiates the transition to next state Sn.
    
    Each triplet (S,A,Sn) identifies the reward value to return.
    
    In a deterministic environment, for any (S,A,Sn) the reward is a constant float value.
    
    In a stochastic environment, (S,A,Sn) can have any number of reward values.
    Reward objects can give constant, weighted tabular, or function-based float reward values.
    """
        
    def __init__(self, const=0.0, reward_probL=None, reward_dist_func=None):
        """
        Reward can be: constant, weighted tabular, or function-based float reward values 
        
        const - constant float
        reward_probL - of (r,w) pairs where r=reward float, w=weight (un-normalized is OK)
        reward_dist_func - a function that returns a range of values (e.g. Gaussian distribution)
        """
        
        self.const = float( const )
        self.reward_probL = reward_probL
        self.reward_dist_func = reward_dist_func
        
        # set flag for Reward type
        if not reward_probL is None:
            self.reward_type = TABLE
        elif not reward_dist_func is None:
            self.reward_type = FUNC
        else:
            self.reward_type = CONST
    
    def __call__(self):
        
        if self.reward_type == CONST:
            return self.const
        elif self.reward_type == TABLE:
            r,w = select_weighted_random( self.reward_probL )
            return r
        elif self.reward_type == FUNC:
            return self.reward_dist_func()
        
    
    def __str__(self):
        
        if self.reward_type == CONST:
            return '<Reward-Constant = %g>'%self.const
        elif self.reward_type == TABLE:
            return '<Reward-Tabular = %s>'%str( self.reward_probL )
        elif self.reward_type == FUNC:
            return '<Reward-Function = %s>'%str( self.reward_dist_func.__name__ )


if __name__ == "__main__": # pragma: no cover
    import random
    
    rc = Reward(const=1.1, reward_probL=None, reward_dist_func=None)
    print( rc )
    
    reward_probL = [(0.0,1), (1.0,1), (2.0,2)] # will be normalized in use.
    rt = Reward(const=1.1, reward_probL=reward_probL, reward_dist_func=None)
    print( rt )
    
    def my_gauss():
        return random.gauss(3.0, 0.5)
    rf = Reward(const=1.1, reward_probL=None, reward_dist_func=my_gauss)
    print( rf )
    
    for r in (rc, rt, rf):
        for i in range(10):
            print(r(), end=' ')
        print()
    
    