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
import numpy as np
from introrl.continuous_sims.param_continuous import ContinuousParameter
from introrl.continuous_sims.sim_continuous import ContinuousSimulation

class MountainCar( ContinuousSimulation ):
    """
    A Simulation of continuous floating point variables
    This Baseline Simulation is for a classic mountain car.
    
    see: for deep learning policy: https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2
    """

    def __init__(self, name='Mountain Car', step_reward=-1.0):
        
        ContinuousSimulation.__init__(self, name=name, step_reward=step_reward)
    # =============== OVERRIDE STARTING HERE =========================
    
    def get_pseudo_coord(self, s_vector, y_scale=1.0, v_scale=1.0):
        """Given a state vector, calc pseudo coordinates for display."""
        
        x,vel = s_vector
        
        # calc y for velocity == 0.0
        y_vzero = self.get_y_pos( x=x )
        
        # offset y up or down depending on velocity
        y = y_vzero * y_scale + vel * v_scale
        return x,y
        
    
    # =============== OVERRIDE ENDING HERE =========================
        

if __name__=="__main__":
    
    MCar = MountainCar( name='Mountain Car', step_reward=-1.0)
    
    MCar.summ_print()
    print('%8.4f x %8.4f xdot'%MCar.get_s_tuple() )
    
    
