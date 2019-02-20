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

DEALER_HITS_BELOW = 17

class Hand( object ):
    """Either the dealer or player hand"""
    def __init__(self, name='???', auto_fill_to_12=False):
        self.name = name
        self.new_hand(auto_fill_to_12=auto_fill_to_12)
        
    def new_hand(self, auto_fill_to_12=False): # can call directly for a new hand
        self.cardL = []
        self.usable_ace = False
        self.get_card(2)
        
        if auto_fill_to_12:
            while self.summ < 11:
                self.get_card(1)

    def get_card(self, N=1):
        for i in range(N):
            self.cardL.append( random.choice( (1,2,3,4,5,6,7,8,9,10,10,10,10) ) )
        self.calc_summ()

    def get_show_card(self):
        return self.cardL[0] # use 1st card as show card.

    def set_player_hand(self, summ=0, usable_ace=False ):
        if usable_ace:
            self.cardL = [1, summ-11]
        else:
            c1 = random.choice( tuple( range(2, summ-1, 1) ) )
            c2 = summ - c1
            self.cardL = [c1, c2]
        
        self.calc_summ()
        
    def set_dealer_hand(self, dealer_showing=0 ):
        self.cardL = [dealer_showing]
        self.get_card(1)

    def calc_summ(self):
        self.usable_ace = False
        
        self.summ = sum( self.cardL )
        if 1 in self.cardL:
            if (self.summ <= 11):
                self.summ += 10 # use the ace as an 11
                self.usable_ace = True
                
        self.is_bust = (self.summ > 21)
        
        self.is_natural = (len(self.cardL)==2) and (self.summ==21)
        
    def summ_print(self):
        s = ''
        if self.is_bust:
            s = '  BUSTED'*4
        elif self.is_natural:
            s = '  NATURAL 21'
        elif self.usable_ace:
            s = '  USABLE ACE'
            
        print('___%s Hand Summary___'%self.name)
        print('    Cards = ', ', '.join( ['%s'%i for i in self.cardL] ))
        print('    Summ  =', self.summ, s)

class BlackJack( object ):
    
    
    def __init__(self):
        self.deal_hands()
    
    def deal_hands(self):
        self.playerH = Hand('Player', auto_fill_to_12=True)
        self.dealerH = Hand('Dealer')
        self.game_over = False
        self.dealer_is_done = False
        self.reward = 0.0
        self.evaluate()
    
    def re_deal(self):
        self.playerH.new_hand( auto_fill_to_12=True )
        self.dealerH.new_hand()
        self.game_over = False
        self.dealer_is_done = False
        self.reward = 0.0
        self.evaluate()
    
    def evaluate(self):
        """
        Only evaluates the player move.  
        Final reward happens when dealer plays.
        """
        if self.playerH.is_bust:
            self.reward = -1.0
            self.game_over = True
            
        elif self.dealerH.is_bust:
            self.reward = 1.0
            self.game_over = True
            
        elif self.playerH.is_natural:
            self.game_over = True
            
            if self.dealerH.is_natural:
                self.reward = 0.0
            else:
                self.reward = 1.0 # player wins outright.
            
    def player_hits(self):
        self.playerH.get_card()
        self.evaluate()
    
    def dealer_plays(self):
        
        while (not self.game_over) and (self.dealerH.summ < DEALER_HITS_BELOW):
            #print('--> Dealer Hits at:', self.dealerH.summ, end=' ')
            self.dealerH.get_card()
            self.evaluate()
            #print('Card is:', self.dealerH.cardL[-1], 'game_over =',self.game_over)
        
        self.dealer_is_done = True
        
        # need to decide winner based on sum of cards
        if not self.game_over: # i.e. nobody busted, so need to compare sums.
            self.game_over = True
            
            if self.dealerH.summ > self.playerH.summ:
                self.reward = -1.0
            elif self.dealerH.summ < self.playerH.summ:
                self.reward = 1.0
            else:
                self.reward = 0.0
    
    def get_state_hash(self):
        """State Hash is (player sum, usable ace flage, dealer show card)"""
        return (self.playerH.summ, self.playerH.usable_ace, self.dealerH.get_show_card() )
    
    def set_state_hash(self, state_hash ):
        (player_sum, usable_ace, dealer_showing) = state_hash
        
        self.playerH.set_player_hand( summ=player_sum, usable_ace=usable_ace )
        self.dealerH.set_dealer_hand( dealer_showing=dealer_showing )
        
        self.game_over = False
        self.dealer_is_done = False
        self.reward = 0.0
        self.evaluate()
    
    def summ_print(self):
        print('___BlackJack___  state_hash =', self.get_state_hash() )
        if self.game_over:
            print('***GAME OVER*** Reward=',self.reward, end=' ')
        
        if self.reward > 0.1:
            print('PLAYER WINS')
        elif self.reward < -0.1:
            print('PLAYER LOSES')
        else:
            print('DRAW GAME')
        
        self.playerH.summ_print()
        self.dealerH.summ_print()

def play_policy_game( BJ, policyD ):
    
    if not BJ.game_over:
        action = policyD[ BJ.get_state_hash() ]
        
        while action == 'Hit':
            #print('--> Player Hits at:', BJ.playerH.summ, end=' ')
            BJ.player_hits()
            #print('Card is:', BJ.playerH.cardL[-1], 
            #      'Hand is:',BJ.playerH.cardL,
            #      '  game_over=',BJ.game_over)
            
            if BJ.game_over:
                action = 'Stay'
            else:
                if BJ.get_state_hash() not in policyD:
                    print('ERROR... BJ.get_state_hash() =',BJ.get_state_hash())
                action = policyD[ BJ.get_state_hash() ]
            
    if not BJ.game_over:
        BJ.dealer_plays()


def mock_game( BJ ):
    if not BJ.game_over:
        if BJ.playerH.summ < 17:
            print('--> Player Hits at:', BJ.playerH.summ, end=' ')
            BJ.player_hits()
            print('Card is:', BJ.playerH.cardL[-1], '  game_over=',BJ.game_over)
    
    if not BJ.game_over:
        BJ.dealer_plays()

if __name__ == "__main__":
    
    BJ = BlackJack()
    mock_game( BJ )
    BJ.summ_print()
    print('-'*55)
    
    BJ.re_deal()
    mock_game( BJ )
    BJ.summ_print()
    print('-'*55)
    
    BJ.set_state_hash( (11, False, 5) )
    mock_game( BJ )
    BJ.summ_print()
    