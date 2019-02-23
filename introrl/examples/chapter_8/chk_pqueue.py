#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import matplotlib
import matplotlib.pyplot as plt
import random
import sys

from introrl.mdp_data.sutton_dyna_grid_xN import get_gridworld

from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.policy import Policy
from introrl.utils.running_ave import RunningAve
from introrl.agents.dyna_q_agent import DynaQAgent
from introrl.agents.priority_sweep_agent import PrioritySweepAgent

ALPHA = 0.5
GAMMA = 0.95
EPSILON=0.1
PLAN_LOOPS = 5
PRIORITY_THRESHOLD=0.0001

NUM_RUNS = 10

q_raveL = []
qp_raveL = []

NUM_SIZES = 3
sum_dynaL   = [0] * NUM_SIZES
sum_psweepL = [0] * NUM_SIZES
grid_sizeL = [0] * NUM_SIZES

def get_greedy_path_length( agent, gridworld, show_path=False ):
    state_pathL = [ gridworld.start_state_hash ]
    Nstates = len(gridworld.SC)
    
    s = state_pathL[-1]
    while len(state_pathL) < Nstates:
        a = agent.action_value_coll.get_best_greedy_action( s )
        s, _ = gridworld.get_action_snext_reward( s, a )
        state_pathL.append( s )
        
        if s in gridworld.terminal_set:
            break
    
    if show_path:
        print( state_pathL )
    
    
    return len( state_pathL )

i_sizes = 2

gridworld = get_gridworld( N_mult=i_sizes+1, step_reward=-PRIORITY_THRESHOLD/(10+i_sizes*10) )
Nstates = len(gridworld.SC)
print('Nstates =',Nstates)
grid_sizeL[i_sizes] = Nstates

for main_loop in range(NUM_RUNS):
    print('%i of %i Runs'%(1+main_loop, NUM_RUNS), end=' ')


    learn_tracker_q = LearnTracker()
    learn_tracker_sw = LearnTracker()


    agent_sw = PrioritySweepAgent( environment=gridworld, 
                     learn_tracker=learn_tracker_sw,
                     max_episode_steps=60000,
                     show_banner = False, do_summ_print=False, show_last_change=False, 
                     epsilon=EPSILON,
                     gamma=GAMMA,
                     alpha=ALPHA, priority_threshold=PRIORITY_THRESHOLD)

    #agent_sw.action_value_coll.summ_print()
    #sys.exit()
    
    # PrioritySweepAgent episodes 
    len_path = float('inf')
    while len_path > gridworld.optimal_path_len:                
        agent_sw.run_episode( gridworld.start_state_hash, Nplanning_loops=PLAN_LOOPS)
        len_path = get_greedy_path_length( agent_sw, gridworld )
    
    print( sum(agent_sw.learn_tracker.steps_per_episode()), ' priority sweep agent')

    sum_psweepL[ i_sizes ] += sum(agent_sw.learn_tracker.steps_per_episode())
    
sum_psweepL[ i_sizes ] /= NUM_RUNS
    
# make all start at zero

print( sum_psweepL )

