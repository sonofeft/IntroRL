#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#colorL = ['r','g','b','c','m','y']
colorL = ['b','r','g','c','m','y']

def plot_policy( updater, Ngrid=100, title='', do_show=True,
                 x2_dataL=None, y2_dataL=None, y2_label='Second Y Axis'):
    
    ff = updater.feature_func
    
    Nactions = len( ff.full_actionL )
    xDL = {} # dict of lists, index=a_desc, value=list of (x,y) tuples
    yDL = {} # dict of lists, index=a_desc, value=list of (x,y) tuples
    for a in ff.full_actionL:
        xDL[a] = []
        yDL[a] = []
    
    x_param = ff.paramL[0]
    y_param = ff.paramL[1]
    
    dx = (x_param.max_value - x_param.min_value) / float(Ngrid)
    dy = (y_param.max_value - y_param.min_value) / float(Ngrid)
    
    s_vector = np.zeros(2)
    
    for i in range(Ngrid+1):
        x = x_param.min_value + i*dx
        s_vector[0] = x
        for j in range(Ngrid+1):
            y = y_param.min_value + j*dy
            s_vector[1] = y
            
            a_best, Qsa = updater.get_max_Qsa( s_vector )
            if a_best is not None:
                xDL[a_best].append( x )
                yDL[a_best].append( y )

    
    fig, ax = plt.subplots()
    plt.xlabel( x_param.name )
    plt.ylabel( y_param.name )
    plt.title( title )
    
    for n in range( Nactions ):
        a = ff.full_actionL[n]
        xL = xDL[a]
        yL = yDL[a]
        color = colorL[ n%len(colorL) ]
        plt.plot(xL, yL, '%ss'%color, label='action=%s'%a)
    
    
    if y2_dataL is not None:
        axr = ax.twinx()
        axr.set_ylabel(y2_label)
        axr.plot( x2_dataL, y2_dataL, 'k-', label=y2_label, linewidth=3)
        axr.legend(loc='best')
        
    ax.legend(loc='best')
    plt.grid()
    fig.tight_layout()
    
    if do_show:
        plt.show()
        
    return fig
    

def plot_cost_to_go( updater, Ngrid=100, do_show=True ):
    """3D plot of max_a( Q(s,a) )"""
    
    ff = updater.feature_func
    
    Nactions = len( ff.full_actionL )
    
    x_param = ff.paramL[0]
    y_param = ff.paramL[1]
    
    dx = (x_param.max_value - x_param.min_value) / float(Ngrid)
    dy = (y_param.max_value - y_param.min_value) / float(Ngrid)
    s_vector = np.zeros(2)
    

    
    fig = plt.figure( figsize=(8,6) )
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.linspace(x_param.min_value, x_param.max_value, num=Ngrid)
    Y = np.linspace(y_param.min_value, y_param.max_value, num=Ngrid)
    Z = []
    for y in  Y:
        rowL = []
        for x in X:
            s_vector[0] = x
            s_vector[1] = y
            a_best, Qsa = updater.get_max_Qsa( s_vector )
            if not Qsa is None:
                rowL.append( -Qsa )
            else:
                rowL.append( 0.0 )
        Z.append( rowL )

    X, Y = np.meshgrid(X, Y)
    Z = np.array( Z )

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    
    # ----------------
    ax.view_init( elev=45.0, azim=-65.0)

    ax.set_xlabel(x_param.name)
    ax.set_ylabel(y_param.name)
    ax.set_zlabel('-max_a[ Q(s,a) ]')
    
    if do_show:
        plt.show()
        
    return fig
    
    
if __name__ == "__main__":  # pragma: no cover
    
    from introrl.continuous_sims.sim_continuous import ContinuousSimulation
    from introrl.continuous_sims.feature_func import FeatureFunction
    from introrl.continuous_sims.feature_func_polynomial import FFPolynomial
    from introrl.continuous_sims.feat_func_tiles import FeatFuncTiles
    from introrl.continuous_sims.update_w_vector import UpdateWVector
        
    sim = ContinuousSimulation(name='Mountain Car', step_reward=-1.0)
        
    #ff = FeatureFunction( sim, name='Proportional', init_w_val=0.0)
    #ff = FFPolynomial(sim, name='Polynomial', init_w_val=0.0, n_degree=2, interaction_only=False)
    ff = FeatFuncTiles(sim, name='TilingsInf', init_w_val=None, num_tiles=8, recenter=True, num_regionsL=[8,8])
    
    ff.init_from_pickle_file( 'mcar_' + ff.desc() )
        
    updater = UpdateWVector( ff )
    
    x_param = ff.paramL[0]
    x2_dataL = np.linspace(x_param.min_value, x_param.max_value, num=100)
    y2_dataL = np.sin(3 * x2_dataL)


    fig1 = plot_policy( updater, Ngrid=100, do_show=False, title='mcar_' + ff.desc(),
                        x2_dataL=x2_dataL, y2_dataL=y2_dataL, y2_label='Height' )
    
    fig2 = plot_cost_to_go( updater, Ngrid=100, do_show=True )
    
    fig1.savefig( 'mcar_policy_' + ff.desc() + ".png")
    fig2.savefig( 'mcar_maxq_' + ff.desc() + ".png")
    
    