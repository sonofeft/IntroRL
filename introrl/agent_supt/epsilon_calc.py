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

class EpsilonGreedy( object ):
    """Make Epsilon-Greedy choices from list of possible actions."""
    
    def __init__(self, epsilon=0.1, const_epsilon=True, half_life=200,
                 N_episodes_wo_decay=0, greedy_mode=False):
        """
        When epsilon is small, more greedy choices are made.
        When epsilon is large, more random choices are made.
        """
        self.epsilon = epsilon
        self.const_epsilon = const_epsilon
        self.half_life = half_life
        self.decay_factor = 1.0 / float( half_life )
        
        self.greedy_mode = greedy_mode # when True, make only greedy choices and do NOT inc. N_eg_choices
        
        self.N_episodes_wo_decay = N_episodes_wo_decay # may wait # of steps before changing eps.
        
        self.N_episodes = 0 # number of episodes (increment with call to inc_N_episodes)
    
        self.N_eg_choices = 0 # may count actions and change epsilon
        self.N_greedy = 0 # track number of greedy and random choices made.
        self.N_random = 0
    
    def set_half_life_for_N_episodes(self, Nepisodes=1000, epsilon_final=0.01):
        """
        Set half_life for a give total number of episodes and a final epsilon value.
        ALSO, set const_epsilon flag to False... assume user's intent.
        """
        half_life = Nepisodes / (self.epsilon/epsilon_final - 1.0)
        self.half_life = half_life
        self.decay_factor = 1.0 / float( half_life )
        self.const_epsilon = False
    
    def set_const_epsilon(self, epsilon_inp=0.1):
        self.epsilon = epsilon_inp
        self.const_epsilon = True
        
    def inc_N_episodes(self): # normally called by Environment
        self.N_episodes += 1
    
    def set_greedy_mode(self, mode=True):
        self.greedy_mode = mode
    
    def greedy_choice(self, greedy_action, legal_actionL ):
        """Make greedy choice, but do not affect epsilon decay."""
        return greedy_action
    
    def current_eps(self):
        """If constant, simply return the constant.  If not, calc the current value."""
        if self.const_epsilon or (self.N_episodes <= self.N_episodes_wo_decay):
            eps = self.epsilon
        else:
            eps = self.epsilon / (1.0 + max(0.0,self.decay_factor * (self.N_episodes-self.N_episodes_wo_decay)))
        return eps        
    
    def __call__(self, greedy_action, legal_actionL, epsilon_inp=None ):
        """
        Return eps-greedy action. 
        Either greedy_action or random pick from legal_actionL.
        """
        
        # If in greedy_mode, always return greedy choice w/o incrementing any counters.
        if self.greedy_mode:
            return greedy_action
        
        self.N_eg_choices += 1
        
        # see if caller wants to dictate epsilon
        if epsilon_inp is None:
            eps = self.current_eps()
        else:
            eps = epsilon_inp

        if random.random() > eps:
            action = greedy_action
            self.N_greedy += 1
            #print('    Made Greedy Choice #%i'%self.N_greedy)
        else:
            action = random.choice( legal_actionL )
            self.N_random += 1
            #print('    Made Epsilon Exploration Choice #%i'%self.N_random)
        
        #print('eps-greedy action =',action)
        return action

    def summ_print(self): # pragma: no cover
        print('___ Epsilon Greedy Summary ___')        
        if self.greedy_mode:
            print('    Currently in Greedy Mode... Always Returns Best Action.')
            return
        
        if self.const_epsilon:
            print('    Constant Epsilon =', self.epsilon)
        else:
            print('       Starting Epsilon =', self.epsilon)
            print('     Episodes w/o Decay =', self.N_episodes_wo_decay)
            print('       Epsilon Halflife =', self.half_life)
            
            if self.const_epsilon or (self.N_episodes <= self.N_episodes_wo_decay):
                eps = self.epsilon
            else:
                eps = self.epsilon / (1.0 + self.decay_factor * (self.N_episodes-self.N_episodes_wo_decay)  )
            
            print('        Current Epsilon =', eps )

        if self.N_episodes:
            print('     # Episodes =', self.N_episodes)
            print('     -----------------------')
            print('     # EpsGreedy Choices =', self.N_eg_choices)
            print('     #    Greedy Choices =', self.N_greedy)
            print('     #    Random Choices =', self.N_random)


if __name__ == "__main__": # pragma: no cover
    
    eg = EpsilonGreedy(epsilon=0.4, const_epsilon=True, half_life=200,
                       N_episodes_wo_decay=0)
    
    for i in range(100):
        print( eg('.', ['.','U','D','L']), end='' )
        eg.inc_N_episodes() # normally called by Environment
    print()
    eg.summ_print()
    print('-'*55)
    eg = EpsilonGreedy(epsilon=0.4, const_epsilon=False, half_life=10,
                       N_episodes_wo_decay=50)
    
    for i in range(100):
        print( eg('.', ['.','U','D','L']), end='' )
        eg.inc_N_episodes() # normally called by Environment
    print()
    eg.summ_print()
    print('Greedy Choice =', eg.greedy_choice('L', ['U','D','L']) )
    
    print('-'*55)
    eg = EpsilonGreedy(epsilon=0.4, const_epsilon=False, half_life=50,
                       N_episodes_wo_decay=50)
    eg.set_greedy_mode( mode=True )
    for i in range(100):
        print( eg('.', ['.','U','D','L']), end='' )
    print()
    eg.summ_print()
    print('Greedy Choice =', eg.greedy_choice('.', ['.','U','D','L']) )

        
