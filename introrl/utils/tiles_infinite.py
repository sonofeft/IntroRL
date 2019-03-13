#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from math import floor, log10, ceil
import numpy as np
from itertools import product

"""
Build tilings for RL application.
"""

class Dimension( object ):
    
    def __init__(self, lo_val=0.0, hi_val=1.0, num_regions=10):
        """
        Break lo to hi val into num_regions.
        NOTE: <lo_val is reigion 0, >=hi_val is num_regions-1
        The number of bounded regions is therefore num_regions-2
        """
        self.lo_val = lo_val
        self.hi_val = hi_val
        self.num_regions = num_regions
        self.max_region = num_regions - 1
        
        if num_regions <= 2:
            raise ValueError( 'num_regions MUST be > 2. Input value =',num_regions )
            
        elif num_regions > 2:
            self.step = float(hi_val - lo_val) / (num_regions-2)
        
    def get_region(self, val):
        """region 0 is < lo_val, region num_regions-1 is >= hi_val."""
        i = floor( (val-self.lo_val) / self.step ) + 1
        return max(0, min(i, self.max_region))
        
    def get_numpy_encoding(self, val):
        """Return a numpy array with 1 in detected regions, 0 elsewhere."""
        encoding = np.zeros( self.num_regions )
        i = self.get_region( val )
        encoding[i ] = 1
        return encoding
        
    def get_nominal_value(self, irange):
        """Return the nominal float value for region number"""
        return self.lo_val - self.step/2.0 + irange*self.step
    
    def summ_print(self):
        print('=============== PartitionedSegment Summary ===============')
        print('    lo_val      =', self.lo_val)
        print('    hi_val      =', self.hi_val)
        print('    num_regions =', self.num_regions)
        print('    step        =', self.step)
        print('    edgeL       =', self.edgeL)

class Tile( object ):
    """
    Some number of Dimension objects to define k dimensional space.
    """
    def __init__(self, lo_valL=None, hi_valL=None, num_regionsL=None):
        """
        Each input is a list defining the lo_val, hi_val and num_regions of
        each dimension in the Tile.
        """
        
        if len(lo_valL)!=len(hi_valL) or len(lo_valL)!=len(num_regionsL):
            raise ValueError( 'All input lists must be same length, '+\
                              'lo_valL, hi_valL and num_regionsL' )
        self.num_dim = len( lo_valL )
        self.dimL = []
        self.np_dim = 0 # track dimension of numpy array for encoding
        self.num_states = 1 # number of possible state positions
        
        for lo_val, hi_val, num_regions in zip(lo_valL, hi_valL, num_regionsL):
            self.dimL.append( Dimension( lo_val=lo_val, hi_val=hi_val, num_regions=num_regions) )
            
            self.np_dim += num_regions
            self.num_states *= num_regions # number of possible state positions
        
        self.state_indexD = {} # index=i_state, value=(i_s1, i_s2, i_s3, ..., i_sn)
        self.rev_state_indexD = {} # index=(i_s1, i_s2, i_s3, ..., i_sn), value=i_state
        self.init_state_index()
        
    def get_regions(self, valL):
        """region 0 is < lo_val, region num_regions-1 is >= hi_val."""
        
        if len(valL) != self.num_dim:
            raise ValueError('Called Tile.get_regions with wrong number of inputs.' +\
                             'length required=%i, valL='%self.num_dim, valL )
        
        return [d.get_region(val) for d,val in zip(self.dimL,valL)]
        
    def get_numpy_encoding(self, valL):
        """Return a numpy array with 1 in detected regions, 0 elsewhere."""
        encoding = np.zeros( self.np_dim )
        iL = self.get_regions(valL)
        i = 0
        for id,d in enumerate(self.dimL):
            encoding[i + iL[id]] = 1
            i += d.num_regions
        return encoding
        
    def get_state_index(self, valL):
        """Get index for position on Tile"""
        return self.rev_state_indexD[ tuple(self.get_regions(valL)) ]
        
    def init_state_index(self):
        rangeL = [range(d.num_regions) for d in self.dimL ]
        for i,tup in enumerate(product(*rangeL)):
            self.state_indexD[i] = tup
            self.rev_state_indexD[tup] = i

    def get_nominal_s_vector(self, i_state):
        """Return the nominal s_vector for the i_state."""
        return [d.get_nominal_value(irange) for d,irange in zip(self.dimL, self.state_indexD[i_state])]

# Calc Log base 2 
def Log2(x): 
    return (log10(x) / log10(2)); 
  
# Function to check if x is power of 2 
def isPowerOfTwo(n): 
    return (ceil(Log2(n)) == floor(Log2(n))); 

class Tilings( object ):
    """
    A number of overlapping Tile objects.
    """
    def __init__(self, home_tile, num_tiles=4, recenter=True, show_pow2_warning=True):
        """
        Build overlapping tiles based off of the home_tile.
        """
        self.home_tile = home_tile
        self.num_tiles = num_tiles
        self.np_dim = home_tile.np_dim * num_tiles # track dimension of numpy array for encoding
        
        if not isPowerOfTwo( num_tiles ):
            print('='*66)
            print('WARNING... Tilings.num_tiles=%i... NOT A POWER of 2.'%num_tiles)
            print('Tilings works best with num_tiles equal to a power of 2 such as 2,4,8,16,etc.')
            print('='*66)
        
        offsetL = [d.step/num_tiles for d in home_tile.dimL]
        #print('offsetL =',offsetL)
        #print('stepL =',[d.step for d in home_tile.dimL],
        #      '   num_regionsL =',[d.num_regions for d in home_tile.dimL])
        
        # build ranges of Tile objects.
        oddL = list( range(1, 2*home_tile.num_dim, 2) )
        addL = oddL[:]
        #print('oddL =', oddL)
        
        lo_valLL = [ [d.lo_val for d in home_tile.dimL] ] # list of lists
        hi_valLL = [ [d.hi_val for d in home_tile.dimL] ]
        
        for _ in range(num_tiles-1):
            
            lo_valL=[]
            hi_valL=[]
            num_regionsL=[]
            for i,d in enumerate(home_tile.dimL):
                lo_valL.append( d.lo_val + addL[i]*offsetL[i] )
                hi_valL.append( d.hi_val + addL[i]*offsetL[i] )
                num_regionsL.append( d.num_regions )
            
            lo_valLL.append( lo_valL )
            hi_valLL.append( hi_valL )
            addL = [(add+odd) % num_tiles for add,odd in zip(addL,oddL)]
        
        if recenter:
            # shift limits of each tile to same center as home_tile
            sumL = [0.0] * home_tile.num_dim
            for lo_valL in lo_valLL:
                for i,v in enumerate( lo_valL ):
                    sumL[i] += v / float( num_tiles )
                    
            diffL = [ s-d.lo_val for s,d in zip(sumL, home_tile.dimL) ]
            #print('diffL =',diffL)
            
            for lo_valL, hi_valL in zip( lo_valLL, hi_valLL ):
                for i,diff in enumerate(diffL):
                    lo_valL[i] = lo_valL[i] - diff 
                    hi_valL[i] = hi_valL[i] - diff 
        
        # build the rest of the tiles.
        self.tileL = []
        for lo_valL, hi_valL in zip( lo_valLL, hi_valLL ):
            self.tileL.append( Tile( lo_valL=lo_valL, hi_valL=hi_valL, num_regionsL=num_regionsL) )
        
    def get_regions(self, valL):
        """region 0 is < lo_val, region num_regions-1 is >= hi_val."""
        
        if len(valL) != self.home_tile.num_dim:
            raise ValueError('Called Tilings.get_regions with wrong number of inputs.' +\
                             'length required=%i, valL='%self.home_tile.num_dim, valL )
        regionL = []
        for tile in self.tileL:
            regionL.extend( tile.get_regions(valL) )
        return regionL
        
    def get_numpy_encoding(self, valL):
        """Return a numpy array with 1 in detected regions, 0 elsewhere."""
        encoding = np.zeros( self.np_dim )
        i = 0
        for tile in self.tileL:
            iL = tile.get_regions(valL)
            for id,d in enumerate(tile.dimL):
                encoding[i + iL[id]] = 1
                i += d.num_regions
        return encoding
        
    def get_state_index(self, valL):
        """Get index for position on Tilings"""
        i_stateT = tuple( [tile.get_state_index(valL) for tile in self.tileL] )
        return i_stateT

            
def plot_tile(ax, tile, color='r', linestyle='-', n=0 ):
    
    
    # only plot 1st 2 dimensions
    d = tile.dimL[n]
    x = d.lo_val
    for i in range(d.num_regions-1):
        line = ax.axvline(x=x, color=color, linestyle=linestyle)
        #print('x=',x, color, linestyle)
        x += d.step
        
    if len(tile.dimL) > 1:
        d = tile.dimL[n+1]
        y = d.lo_val
        for i in range(d.num_regions-1):
            line = ax.axhline(y=y, color=color, linestyle=linestyle)
            #print('y=',y, color, linestyle)
            y += d.step
            

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    d = Dimension(lo_val=0.0, hi_val=1.0, num_regions=4)
    if 1:
        for val in [-99,-.1, 0.0, .1, 0.5, .9, 1.0, 1.1]:
            print('%8g %i'%(val, d.get_region(val)), d.get_numpy_encoding(val))
    print('-'*66)
    
    t = Tile(lo_valL=[0,0], hi_valL=[1,10], num_regionsL=[4,5])
    print('t.state_indexD =',t.state_indexD)
    if 1:
        for valL in [(-99,-99), (-1,11),(0,0), (.1,2), (.9,9), (99,99)]:
            print('%s %s'%(valL, t.get_regions(valL)), end=' ')
            print(t.get_numpy_encoding(valL))
        
    print('='*66)
    T = Tilings( t, num_tiles=4 )
    if 1:
        for valL in [(-99,-99), (-1,11),(0,0), (.1,2), (.9,9), (99,99)]:
            print('%s %s'%(valL, T.get_regions(valL)), end=' ')
            print(T.get_numpy_encoding(valL))
    
    if 1:
        fig, ax = plt.subplots()
        n=0

        colorL = ['r','g','b','c','m','y']
        linestyleL = ['-',':','--']
        
        for itile, tile in enumerate(T.tileL):
            print(itile,'Tile')
            plot_tile(ax, tile, color=colorL[itile%len(colorL)], 
                      linestyle=linestyleL[itile%len(linestyleL)], n=n )

        plt.show()

    
    