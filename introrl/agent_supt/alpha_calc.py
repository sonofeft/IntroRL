#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

class Alpha( object ):
    """Learning rate for Learner objects. Can be constant alpha or decay based on different rules."""
    
    def __init__(self, alpha=0.1, const_alpha=True, half_life=200, N_episodes_wo_decay=0):
        """
        If const_alpha set to True, use a constant alpha.
        Otherwise, track the number of calls for given states and actions and decay by half_life.
        """
        
        self.alpha = alpha
        self.const_alpha = const_alpha
        self.half_life = half_life
        self.decay_factor = 1.0 / float( half_life )
        self.N_episodes_wo_decay = N_episodes_wo_decay
        self.N_episodes = 0 # number of episodes (increment with call to inc_N_episodes)
                
    def __call__(self):
        """Return current learning rate alpha with any applicable decay."""
        
        if self.const_alpha:
            return self.alpha
        
        alpha = self.alpha / (1.0 + max(0.0,self.decay_factor * (self.N_episodes-self.N_episodes_wo_decay)))
        
        #print(alpha, args, self.decayD)
        return alpha
        
    def inc_N_episodes(self): # normally called by Environment
        self.N_episodes += 1

    def summ_print(self):
        print('___ Alpha Summary ___')        
        if self.const_alpha:
            print('    Constant Alpha =', self.alpha)
        else:
            print('    Starting Alpha  =', self.alpha)
            print('    Alpha Halflife  =', self.half_life)
            print('    N eps. wo decay =', self.N_episodes_wo_decay)
            print('    Current Alpha   =', self() )

            
        
        
 
if __name__ == "__main__":
    
    a = Alpha(alpha=0.1, const_alpha=True, half_life=200)
    print('Basic alpha =', a() )
    a.summ_print()

    print('-'*55)
    a2 = Alpha(alpha=0.1, const_alpha=False, half_life=200, N_episodes_wo_decay=99)
    for _ in range(100):
        a2.inc_N_episodes()
    print('w/o args alpha =', a2() )
    print('w/o args alpha =', a2() )
    print('w/o args alpha =', a2() )
    print('w/o args alpha =', a2() )
    print('w/o args alpha =', a2() )
    print('w/o args alpha =', a2() )
    a2.summ_print()


    print('-'*55)
    a3 = Alpha(alpha=0.1, const_alpha=False, half_life=200)
    for _ in range(400):
        a3.inc_N_episodes()
    print('w/o args alpha =', a3() )
    print('w/o args alpha =', a3() )
    print('w/o args alpha =', a3() )
    print('w/o args alpha =', a3() )
    print('w/o args alpha =', a3() )
    print('w/o args alpha =', a3() )
    a3.summ_print()
