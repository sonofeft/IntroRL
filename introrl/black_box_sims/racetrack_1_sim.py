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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from introrl.black_box_sims.sim_baseline import Simulation
from introrl.black_box_sims.racetrack_legal_moves import legal_moveD


# -------- make layout template for states ---------
s_hash_rowL = [] # layout rows for makeing 2D output
row_tickL = None
col_tickL = None
in_boundsL = []

w_track = 17
h_track = 32
def add_row(irow, jstart, jend ):
    # start by making basic (irow,j) rows (i.e. ignore vx, vy terms for now)
    rowL = []
    for j in range( w_track ):
        if j>=jstart and j<=jend:
            rowL.append( (irow, j) )
            in_boundsL.append( (irow, j) )
        else:
            rowL.append( '' )
    
    # start creating vx and vy iteration loops for full (irow,j, vx,vy) tuples
    for vx in range( 5 ):
        for vy in range( 5 ):
            vrowL = []
            if (irow>0) and (vx==0) and (vy==0):
                pass # disallow zero velocity above the 1st row
            else:
                for r in rowL:
                    if r == '': # ignore empty locations
                        vrowL.append( '' )
                    else:
                        (i,j) = r
                        vrowL.append( (i,j,vx,vy) )
            # if the row is defined, save it to s_hash_rowL
            if vrowL:
                s_hash_rowL.append( vrowL )

for i in range(3):
    add_row(i, 3, 8 )
    
for i in range(3, 10):
    add_row(i, 2, 8 )
    
for i in range(10, 18):
    add_row(i, 1, 8 )
    
for i in range(18, 25):
    add_row(i, 0, 8 )

add_row(25, 0, 9 )
add_row(26, 0, 16 )
add_row(27, 0, 16 )
add_row(28, 1, 16 )
add_row(29, 2, 16 )
add_row(30, 2, 16 )
add_row(31, 3, 16 )
s_hash_rowL.append( ['','','','','','','','','','','','','','','','','Done'] )
s_hash_rowL.reverse() # put start line at bottom of page

racetrack_area = set( in_boundsL )
starting_lineL = [(0,3,0,0), (0,4,0,0), (0,5,0,0), (0,6,0,0), (0,7,0,0), (0,8,0,0)]
finish_lineL = [(26,16), (27,16), (28,16), (29,16), (30,16), (31,16)]

#print('(17, 4) in racetrack_area = ',(17, 4) in racetrack_area)

class RaceTrack_1( Simulation ):
    
    def __init__(self, name='RaceTrack_1 Simulation', s_hash_rowL=s_hash_rowL, 
                 enable_random_zero_deltav=True,
                 row_tickL=row_tickL, 
                 col_tickL=col_tickL, 
                 x_axis_label='Starting Line', 
                 y_axis_label='Finish                         '):
        """
        A Black Box Interface to a Simulation
        """
        Simulation.__init__(self, name=name, s_hash_rowL=s_hash_rowL, 
                            row_tickL=row_tickL, x_axis_label=x_axis_label,
                            y_axis_label=y_axis_label, col_tickL=col_tickL)
        
        self.racetrack_area = racetrack_area # set( [(i1,j1), (i2,j2), ...] )
        self.starting_lineL = starting_lineL # [(x1,y1,0,0), (x2,y2,0,0), ...]
        self.finish_lineL = finish_lineL # [(x1,y1), (x2,y2), ...]
        
        self.start_state_hash = starting_lineL[0]
        
        # if enabled, with prob=0.1, deltav will be set to (0,0)
        self.enable_random_zero_deltav = enable_random_zero_deltav
        
        self.default_policyD = {}
        
        # state hash
        self.action_state_set = set() # a set of action state hashes
        terminalL = [('Done','Done',0,0)] # terminal state hashes.
        
        for i in range(32):
            for j in range(17):
                for vx in range(5):
                    for vy in range(5):
                        self.action_state_set.add( (i,j, vx,vy) )
                        
                        if i<24:
                            self.default_policyD[ (i,j, vx,vy) ] = (4-vx, -vy)
                        else:
                            self.default_policyD[ (i,j, vx,vy) ] = (-vx, 4-vy)
                            
                            
                

        self.terminal_set = set( terminalL )
        
        # make sure all default_policyD entries are legal 
        delete_s_hashL = []
        for s_hash, a_desc in self.default_policyD.items():
            (x,y, vx,vy) = s_hash
            if not (x,y) in self.racetrack_area:
                delete_s_hashL.append( s_hash )
            else:
                aL = self.get_state_legal_action_list( s_hash )
                if a_desc not in aL:
                    if aL:
                        self.default_policyD[ s_hash ] = aL[0]
                        #print('replaced default_policyD["%s"]'%str(s_hash),a_desc,' with ',aL[0])
                    else:
                        delete_s_hashL.append( s_hash )

        for s_hash in delete_s_hashL:
            del self.default_policyD[ s_hash ]


    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        reward = -1 # every step has same reward
        
        #print('s_hash, a_desc =',s_hash, a_desc)
        (x,y, vx,vy) = s_hash
        (dvx, dvy) = a_desc
        
        # random deltav = 0,0
        if (vx,vy) != (0,0):
            if self.enable_random_zero_deltav and random.random()<0.1:
                (dvx, dvy) = (0, 0)
        
        vx2 = max(0, min(4, vx+dvx))
        vy2 = max(0, min(4, vy+dvy))
        x2 = x + vx
        y2 = y + vy
        
        if (x2,y2) in self.racetrack_area:
            #print( (x2,y2), end='' )
            
            sn_hash = (x2, y2, vx2, vy2)
            #if (vx2, vy2)==(0,0):
            #    print('OOPS... all zero velocity')
        else:
            if y2>=16 and (x2>=26 and x2<=31):
                sn_hash = ('Done','Done',0,0)
                #reward = 100 # provide big incentive to cross finish line
                #print('Done', s_hash, a_desc, end='')
            else:
                sn_hash = random.choice( self.starting_lineL )
                #print('OB...', s_hash, sn_hash)
            
        return sn_hash, reward
        
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        (x,y, vx,vy) = s_hash
        
        if x in ['Done','Crash']:
            return [] # [(None,None)]
        
        if not (x,y) in self.racetrack_area:
            #print('off self.racetrack_area', s_hash)
            return [] # [(None,None)]
            
        return legal_moveD[ (vx,vy) ]

    def limited_start_state_list(self):
        """
        Return a limited list of starting states.
        Normally used by agents that need to discover the various
        states in an environment, like epsilon-greedy.
        """
        
        lim_stateL = self.starting_lineL
        return lim_stateL
    
    def plot_policy(self, ax, policy_obj ):
        """Need to create fig and ax from matplotlib externally"""
        
        policyD = policy_obj.make_dict_of_policy()

        for (j,i) in self.racetrack_area:
            rect = mpatches.Rectangle((i-.5, j-.5), 1.0, 1.0, ec="none", color='blue', alpha=0.3)
            ax.add_patch(rect)

        for (j,i,_,_) in self.starting_lineL:
            rect = mpatches.Rectangle((i-.5, j-.5), 1.0, 1.0, ec="none", color='yellow', alpha=1.)
            ax.add_patch(rect)

        for (j,i) in self.finish_lineL:
            rect = mpatches.Rectangle((i-.5, j-.5), 1.0, 1.0, ec="none", color='green', alpha=1.)
            ax.add_patch(rect)

        markersize = len(self.starting_lineL)*2 + 4
        linewidth = len(self.starting_lineL)*2 + 2

        for start_state in self.limited_start_state_list():
            print('For Start State =', start_state)
            
            s_hash = start_state
            (x,y, vx,vy) = s_hash
            xL=[x]
            yL=[y]
            
            
            while s_hash in policyD:
            
                a_desc = policyD[ s_hash ]
                (dvx, dvy) = a_desc
                
                x2 = x + vx
                y2 = y + vy
                vx2 = max(0, min(4, vx+dvx))
                vy2 = max(0, min(4, vy+dvy))
                
                sn_hash = (x2,y2, vx2,vy2)
                print('action ',a_desc,' leads to ',sn_hash)
                s_hash = sn_hash
                (x,y, vx,vy) = s_hash
                xL.append( x )
                yL.append( y )
            
            ax.plot(yL, xL, 'o-', markersize=markersize, linewidth=linewidth)
            markersize -= 2
            linewidth -= 2
            print('-'*55)

        ax.set(title=self.name)
        ax.grid()
        ax.set_aspect(1.0)
        
    def get_policy_score(self, policy=None, start_state_hash=None, step_limit=1000):
    
        r_sum, n_steps = 0, 0
        
        if start_state_hash is None:
            sL = starting_lineL
        else:
            sL = [start_state_hash]
        
        for ss in sL:
            (r,n, msg) = Simulation.get_policy_score(self, policy, start_state_hash=ss, step_limit=step_limit)
            r_sum += r
            n_steps += n
            
        msg = '' # any special message(s)
        return (r_sum, n_steps, msg)

if __name__ == "__main__": # pragma: no cover
    
    import time
    import os, sys
    
    start_time = time.time()
    
    RT = RaceTrack_1()
    
    score = RT.get_policy_score( policy=None, start_state_hash=None, step_limit=1000)
    print('Score for RT is ',score, ' = (r_sum, n_steps, msg)')
    
    #RT.layout.s_hash_print()
    
    print( 'RT.get_state_legal_action_list( (17, 4, 3, 0) ) =',RT.get_state_legal_action_list( (17, 4, 3, 0) ) )
