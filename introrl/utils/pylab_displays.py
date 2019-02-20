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

from introrl.utils.banner import banner

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties

    got_matplotlob = True
except:
    got_matplotlob = False

alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}

def scale_segment( x1, y1, x2, y2, trim_frac=0.2):
    """Scale segment and leave center point of segment constant"""    
    # only allow changing segment within the starting and ending rectangles.
    dx = min(1.0, abs(x2 - x1)) * trim_frac
    dy = min(1.0, abs(y2 - y1)) * trim_frac
    
    if y2>y1:
        Y2 = y2 - dy
        Y1 = y1 + dy
    else:
        Y2 = y2 + dy
        Y1 = y1 - dy
    
    if x2>x1:
        X2 = x2 - dx
        X1 = x1 + dx
    else:
        X2 = x2 + dx
        X1 = x1 - dx
    
    return X1,Y1, X2,Y2
    

def draw_arrow( plt, Nrows, env, s_hash, a_desc, pad, 
                color='r', Rinner=0.25, frac_len=0.25):
    """Draw an arrow from s_hash to sn_hash that comes from taking action a_desc."""
    
    if env.layout is None:
        return
    
    (rs, cs) = env.layout.get_row_col( s_hash )
    if rs is None:
        return
        
    sn_hash, reward = env.get_action_snext_reward( s_hash, a_desc )
    (rsn, csn) = env.layout.get_row_col( sn_hash )
    if rsn is None:
        return
    
    dx = csn - cs 
    dy = rsn - rs 
    angle = math.atan2( dy, dx ) - math.pi / 2.0
    
    xoff = Rinner * math.cos( angle )
    yoff = Rinner * math.sin( angle )
    
    p2 = pad / 2.0
    
    x1 = xoff + cs + 0.5 - p2
    y1 = Nrows - (yoff + rs ) - 0.5 - p2
    x2 = xoff + csn + 0.5 - p2
    y2 = Nrows - (yoff + rsn) - 0.5 - p2
    
    x1,y1,x2,y2 = scale_segment( x1, y1, x2, y2, trim_frac=frac_len)
    
    plt.arrow(x1, y1, x2-x1, y2-y1, fc=color, ec='k', width=0.05, length_includes_head=True)

def plot_grid_numbers( rows_outL, header='', x_axis_label='', do_show=True, fmt='%g'):
    
    if not got_matplotlob:
        banner('ERROR: could not import matplotlib\n"plot_grid_numbers" FAILED.')
        return
        
    Nrows = len(rows_outL)
    Ncols = max( [len(row) for row in rows_outL] )
    
    
    fig, axs = plt.subplots()
    plt.axes()
    
    font = FontProperties()
    font.set_size('large')
    font.set_family('fantasy')
    font.set_style('normal')

    
    for i in range( Nrows ):
        rowL = rows_outL[ i ]
        x = Nrows-i-1
        for j in range( Ncols ):
            if j < len(rowL):
                s = rowL[j]
            else:
                s = '*'
            #      Rectangle(  (x,y),    width,   height)
            rect = Rectangle((j,x), 0.9,   0.9, fc='r', alpha=0.5, edgecolor='black')
            plt.gca().add_patch( rect )
            
            t = plt.text(j+.45, x+.45, s, fontproperties=font,
                 **alignment)

    
    
    plt.xlim(0, Ncols)
    plt.ylim(0, Nrows)

    plt.show()
    

if __name__ == "__main__": # pragma: no cover
    
    plot_grid_numbers([[3.3, 2.2],[5.5, 7.7]])