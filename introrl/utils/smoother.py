#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

'''A simple smoothing algorithm
The following algorithm smoothes data using n-point sliding box-car smoothing. 
'''

def boxcar(xL, n):
    '''
    You can smooth the data data1 using k points by calling the 
    function boxcar(xL, n) where n is HALF the size of boxcar  
    
    # xL is list to be smoothed, n is HALF the size of boxcar 
    # (preserve position, so use points from -n to +n
    '''
    
    x_smooth = []
    L = len(xL)
        
    for i in range( L ):
        if i<n: # 1st n points will have smaller than n smoothing
            imin=0
            imax=i*2+1
        elif i<L-n:# middle of curve will have full n-sized boxcar
            imin = i-n
            imax = i+n
        else: # last n points will have smaller than n smoothing
            imax = L
            imin = i - (L-i) - 1
            
        span = imax-imin
        val = sum( xL[imin:imax] ) / span
        x_smooth.append( val )
        
    return x_smooth


def boxcarMedian(xL, n):
    # xL is list to be smoothed, n is HALF the size of boxcar 
    # (preserve position, so use points from -n to +n
    # Sorts boxcar and returns median as the value
    
    x_smooth = []
    L = len(xL)
        
    for i in range( L ):
        if i<n: # 1st n points will have smaller than n smoothing
            imin=0
            imax=n
        elif i<L-n:# middle of curve will have full n-sized boxcar
            imin = i-n
            imax = i+n
        else: # last n points will have smaller than n smoothing
            imax = L
            imin = L - n
            
        memberL =  xL[imin:imax] 
        memberL.sort()
        k = (imax-imin)/2
        x_smooth.append( memberL[k] )
        
    return x_smooth

def sliceAverage(xL, n):
    # xL is list to be smoothed, n is HALF the size of boxcar 
    # (preserve position IF!!! TIME is also sliceAveraged, so use points from -n to +n
    x_smooth = []
    L = len(xL)
    r = L % n
    L = L-r
        
    for i in range(0, L, n ):
        val = sum( xL[i:i+n] ) / n
        x_smooth.append( val )
        
    return x_smooth
    


if __name__ == "__main__": # pragma: no cover
    
    import matplotlib.pyplot as plt
    import random
    
    xL = [5.0 + random.random() for _ in range(100)]
    smoothL = boxcar(xL, 5)
    
    plt.plot(xL, label='Original')
    plt.plot(smoothL, label='Smoothed')
    plt.legend()
    plt.show()
    
    