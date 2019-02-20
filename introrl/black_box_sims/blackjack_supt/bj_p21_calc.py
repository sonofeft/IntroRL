#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import sys
import os

from introrl.black_box_sims.blackjack_supt.blackjack import BlackJack, play_policy_game

policyD = {}

BJ = BlackJack()

resultsD = {} # index=(player_sum, usable_ace, dealer_showing): 
              # value={'Hit':[win, lose, draw], 'S':[win, lose, draw]}



def run_loop(player_sum=21, action='Hit', usable_ace=False ):
    for loop in range( 100000 ):
        for dealer_showing in range(1, 11, 1):
        
            BJ.re_deal()
            
            BJ.set_state_hash( (player_sum, usable_ace, dealer_showing) )
            state_hash = BJ.get_state_hash() # should be the same
            
            policyD[ state_hash ] = action # set to input action
            
            play_policy_game( BJ, policyD )
            
            if state_hash not in resultsD:
                resultsD[ state_hash ] = {'Hit':[0,0,0], 'S':[0,0,0]}
                
            if BJ.reward > 0.1:
                resultsD[ state_hash ][action][0] += 1 # player wins
            elif BJ.reward < -0.1:
                resultsD[ state_hash ][action][1] += 1 # player loses
            else:
                resultsD[ state_hash ][action][2] += 1 # draw game
            
# Need to run usable_ace=True AFTER running  usable_ace=False
#run_loop( player_sum=21, action='Hit'   , usable_ace=False)
#run_loop( player_sum=21, action='S'   , usable_ace=False)

for usable_ace in [False, True]:
    for player_sum in range(21, 10, -1):
        print( player_sum, end=' ' )
        for action in ['Hit','S']:
            run_loop( player_sum=player_sum, action=action , usable_ace=usable_ace)
print()

# -------- Build bj_policy.py ----------
#fOut = open('bj_policy.py','w')
fOut = sys.stdout

fOut.write("""
resultsD = {} # index=(player_sum, usable_ace, dealer_showing): 
              # value={'Hit':[win, lose, draw], 'S':[win, lose, draw]}

""")
for state_hash in sorted( resultsD.keys() ):
    fOut.write( 'resultsD[%16s] = %s\n'%(str(state_hash), str(resultsD[state_hash])) )

fOut.write("""

# bj_policyD - index=(player_sum, usable_ace, dealer_showing)
bj_policyD = {}

for state_hash in resultsD.keys():
    (player_sum, usable_ace, dealer_showing) = state_hash
    (win_h, lose_h, draw_h) = resultsD[state_hash]['Hit']
    (win_s, lose_s, draw_s) = resultsD[state_hash]['S']
    
    if player_sum==21:
        bj_policyD[ state_hash ] = 'S'
    else:
        if (win_h-lose_h) > (win_s-lose_s):
            bj_policyD[ state_hash ] = 'Hit'
        else:
            bj_policyD[ state_hash ] = 'S'
            

if __name__ == "__main__":
    
    for state_hash in sorted(bj_policyD.keys()):
        print( '%16s'%str(state_hash), bj_policyD[state_hash] )
    

""")


fOut.write('\n\n')
fOut.close()

