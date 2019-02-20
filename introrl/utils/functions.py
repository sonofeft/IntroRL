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


def select_weighted_random( iwL ):
    """
    Return an (i,w) pair 
    from a list of (i,w) pairs where i=anything, w=weight,
    select (i,w) as a weighted probability
    
    FASTER THAN I THOUGHT... returns as soon as value is found.
    (if the input list is presorted from big to small weights, would be fastest.)
    """
    wtot = 0.0
    for (i,w) in iwL: # tried sum([list comprehension]), but was slower
        wtot += w
        
    wr = wtot * random.random()
    for (i,w) in iwL:
        if w > wr:
            return (i,w)
        wr -= w
    return (i,w) # return last value

'''
def weighted_choice(choices):
    #from bisect import bisect # <--- requires bisect
    """TAKES TWICE AS LONG AS select_weighted_random"""
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return choices[i]        


def python3_weighted_choice(choices):
    """TAKES FOUR TIMES AS LONG AS select_weighted_random"""
    values, weights = zip(*choices)
    val = random.choices( values, weights=weights ) # <-- python 3 ONLY
    return val    
'''

def clamp(n, lo, hi):
    """clamp n to a value between lo and hi inclusive."""
    return max(min(hi, n), lo)

def max_val_in_dict( d ):
    if len(d) == 0:
        return 0.0
    else:
        return max( list(d.values()) )

#def argmax_vmax_list_v2(iwL, pick_random_best=True, err_delta=0.0001):
    
#    wiL = sorted( [(w,i) for i,w in iwL] )
    
#    if pick_random_best:
#        actionL = [i for w,i in wiL if w>=wiL[-1][0]-err_delta]
#        return actionL, wiL[-1][0]
#    else:
#        return wiL[1], wiL[0]
    

def argmax_vmax_list(iwL, pick_random_best=True):
    """
    Return an (i,w) pair 
    from a list of (i,w) pairs where i=anything, w=weight,
    select (i,w) where w is the maximum w in the list.
    
    if pick_random_best is True, then decide ties randomly.
    """
    
    # returns the argmax (i) and max (w) from a list
    max_i = None
    max_w = float('-inf')
    maxL = []
    for i,w in iwL:
        if w > max_w:
            max_w = w
            max_i = i
            maxL = [ (i,w) ]
        elif w == max_w:
            maxL.append( (i,w) )
    
    if pick_random_best:
        return random.choice( maxL )
    else:
        return max_i, max_w

def argmax_vmax_dict(d, pick_random_best=True):
    """
    returns the argmax (key) and max (value) from a dictionary    
    if pick_random_best is True, then decide ties randomly.
    """
    
    max_key = None
    max_val = float('-inf')
    maxL = []
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
            maxL = [ (k,v) ]
        elif v == max_val:
            maxL.append( (k,v) )
    
    if pick_random_best:
        return random.choice( maxL )
    else:
        return max_key, max_val

def multi_argmax_vmax_dict( d, err_delta=0.0001 ):
    """
    returns a list of argmax (key) and max (value) from a dictionary    
    All keys within err_delta will be included in list.
    """
    try:
        vkL = sorted( [(v,k) for k,v in d.items()] )
        
        actionL = [k for v,k in vkL if v>=vkL[-1][0]-err_delta]
        return actionL, vkL[-1][0]
    except:
        return [], None
    

def min_val_in_dict( d ):
    if len(d) == 0:
        return 0.0
    else:
        return min( list(d.values()) )

def argmin_vmin_dict(d):
    # returns the argmin (key) and min (value) from a dictionary
    min_key = None
    min_val = float('inf')
    for k, v in d.items():
        if v < min_val:
            min_val = v
            min_key = k
    return min_key, min_val
    



def intCast( val=0 ):
    try:
        return int(val)
    except:
        return 0
        
def floatCast( val=0.0 ):
    try:
        return float(val)
    except:
        return 0.0

def is_int( ival ):
    
    if type(ival)==type(11):
        return 1
    
    if type(ival)==type('string'):
        if ival.find('.')>=0:
            return 0

    try:
        if intCast( ival) == int(ival):
            return 1
    except:
        pass
        
    return 0


def is_float( fval ):
    
    if type(fval)==type(11.11):
        return 1
    
    if type(fval)==type('string'):
        if fval.find('.')==-1:
            return 0

    try:
        if floatCast( fval) == float(fval):
            return 1
    except:
        pass
        
    return 0


if __name__ == "__main__": # pragma: no cover
    
    import timeit
    from bisect import bisect
    
    print('clamp(0, 1, 3) =', clamp(0, 1, 3))
    print('clamp(5, 1, 3) =', clamp(5, 1, 3))
    
    d = {'neg1':-1, 'zero':0, 'pos1':1}
        
    print( 'max_val_in_dict( d )',max_val_in_dict( d ) )
    print( 'min_val_in_dict( d )', min_val_in_dict( d ))
    
    print('argmax_vmax_dict(d)',argmax_vmax_dict(d))
    print('argmin_vmin_dict(d)',argmin_vmin_dict(d))
    
    iwL = [('ONE',.1),('2',.1),('3',.1),('4',.1),('5',.1),('6',.1),('7',.1), (9,.9)]
    iwL_sortL = sorted( iwL, reverse=True, key=lambda _: _[-1] )
    print(iwL_sortL)

    print('---------- argmax_vmax_list vs argmax_vmax_list_v2 ----------')

    argmax_vmax_list_v2

    #print('argmax_vmax_list_v2 timeit results:', timeit.timeit('choice, val = argmax_vmax_list_v2(iwL)', 
    #                                       setup="from __main__ import argmax_vmax_list_v2, iwL") )

    print('argmax_vmax_list timeit results:', timeit.timeit('choice, val = argmax_vmax_list(iwL)', 
                                           setup="from __main__ import argmax_vmax_list, iwL") )

    print('---------- select_weighted_random approach ----------')
    one_count = 0
    NLOOPS = 40
    for loop in range( NLOOPS ):
        choice = select_weighted_random(iwL)
        #print( choice, end=' ' )
        if choice[0] == 'ONE':
            one_count += 1
    print('select_weighted_random approach one_count =', one_count,' = ', float(one_count)/float(NLOOPS))
    print()
    print('select_weighted_random timeit results:', timeit.timeit('choice = select_weighted_random(iwL)', 
                                           setup="from __main__ import select_weighted_random, iwL") )
    
    
    print('---------- reverse sorted select_weighted_random approach ----------')
    one_count = 0
    NLOOPS = 40
    for loop in range( NLOOPS ):
        choice = select_weighted_random(iwL)
        #print( choice, end=' ' )
        if choice[0] == 'ONE':
            one_count += 1
    print('select_weighted_random approach one_count =', one_count,' = ', float(one_count)/float(NLOOPS))
    print()
    print('select_weighted_random timeit results:', timeit.timeit('choice = select_weighted_random(iwL_sortL)', 
                                           setup="from __main__ import select_weighted_random, iwL_sortL") )
    
    

