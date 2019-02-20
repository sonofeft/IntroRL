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

from introrl.utils.grid_funcs import print_string_rows
from introrl.black_box_sims.blackjack_supt.bj_policy import bj_policyD, resultsD
# bj_policyD - index=(player_sum, usable_ace, dealer_showing)

Nruns = sum( resultsD[ (21, False, 10)]['S'] )
print('   *** Charts based on %i runs ***'%Nruns)

# ------------------------

usable_aceL = []
for player_sum in range(21, 11, -1):
    rowL = []
    for dealer_showing in range(1, 11):
        rowL.append( bj_policyD[ (player_sum, True, dealer_showing) ] )
    usable_aceL.append( rowL )

row_tickL=[i for i in range(21, 10, -1)]

print_string_rows( usable_aceL, row_tickL=row_tickL, const_col_w=True, 
                   line_chr='=', left_pad='',
                   header='Usable Ace Policy', 
                   x_axis_label='A  2   3   4   5   6   7   8   9  10 ', 
                   justify='left') # left, right, center
print('              Dealer Showing')
print()

# ------------------------

no_usable_aceL = []
for player_sum in range(21, 10, -1):
    rowL = []
    for dealer_showing in range(1, 11):
        rowL.append( bj_policyD[ (player_sum, False, dealer_showing) ] )
    no_usable_aceL.append( rowL )

row_tickL=[i for i in range(21, 10, -1)]

print_string_rows( no_usable_aceL, row_tickL=row_tickL, const_col_w=True, 
                   line_chr='=', left_pad='',
                   header='No Usable Ace Policy', 
                   x_axis_label='A  2   3   4   5   6   7   8   9  10 ', 
                   justify='left') # left, right, center
print('              Dealer Showing')
        

# ------------------------
