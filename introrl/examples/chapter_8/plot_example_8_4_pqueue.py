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
PLANNING_LOOPS = 5
PRIORITY_THRESHOLD=0.0001

NUM_RUNS = 10

q_raveL = []
qp_raveL = []

NUM_SIZES = 8
sum_dynaL   = [0] * NUM_SIZES
sum_psweepL = [0] * NUM_SIZES

numup_dynaL   = [0] * NUM_SIZES
numup_psweepL = [0] * NUM_SIZES

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

for i_sizes in range( NUM_SIZES ):
    
    N_mult = i_sizes + 1

    gridworld = get_gridworld( N_mult=N_mult) #, step_reward=-PRIORITY_THRESHOLD )
    Nstates = len(gridworld.SC)
    print('Nstates =',Nstates, '  N_mult = ', N_mult)
    grid_sizeL[i_sizes] = N_mult

    for main_loop in range(NUM_RUNS):
        print('%i of %i Runs'%(1+main_loop, NUM_RUNS), end=' ')


        learn_tracker_q = LearnTracker()
        learn_tracker_sw = LearnTracker()

        agent_q = DynaQAgent( environment=gridworld, 
                         learn_tracker=learn_tracker_q,
                         max_episode_steps=60000,
                         show_banner = False, do_summ_print=False, show_last_change=False, 
                         epsilon=EPSILON,
                         gamma=GAMMA,
                         alpha=ALPHA)

        agent_sw = PrioritySweepAgent( environment=gridworld, 
                         learn_tracker=learn_tracker_sw,
                         max_episode_steps=60000,
                         show_banner = False, do_summ_print=False, show_last_change=False, 
                         epsilon=EPSILON,
                         gamma=GAMMA,
                         alpha=ALPHA, priority_threshold=PRIORITY_THRESHOLD)

        # DynaQ episodes 
        len_path = float('inf')
        while len_path > gridworld.optimal_path_len:
            agent_q.run_episode( gridworld.start_state_hash, Nplanning_loops=PLANNING_LOOPS)
            len_path = get_greedy_path_length( agent_q, gridworld )
            
        print(' dyna_q_agent', agent_q.num_updates, end=' ')
        
        # PrioritySweepAgent episodes 
        len_path = float('inf')
        while len_path > gridworld.optimal_path_len:                
            agent_sw.run_episode( gridworld.start_state_hash, Nplanning_loops=PLANNING_LOOPS)
            len_path = get_greedy_path_length( agent_sw, gridworld )
        
        print('vs.', agent_sw.num_updates, ' priority sweep agent')

        sum_dynaL[ i_sizes ]   += sum(agent_q.learn_tracker.steps_per_episode())
        sum_psweepL[ i_sizes ] += sum(agent_sw.learn_tracker.steps_per_episode())
        
        numup_dynaL[ i_sizes ]   += agent_q.num_updates
        numup_psweepL[ i_sizes ]   += agent_sw.num_updates
        
    sum_dynaL[ i_sizes ]   /= NUM_RUNS
    sum_psweepL[ i_sizes ] /= NUM_RUNS
    
    numup_dynaL[ i_sizes ]   /= NUM_RUNS
    numup_psweepL[ i_sizes ] /= NUM_RUNS


fig, ax = plt.subplots()

print('grid_sizeL =', grid_sizeL)

ax.plot(grid_sizeL, numup_dynaL, 'c-', label='Dyna-Q, IntroRL' )
ax.plot(grid_sizeL, numup_psweepL, 'r-', label='PSweep, IntroRL' )
print('numup_dynaL =', numup_dynaL)
print('numup_psweepL =', numup_psweepL)

#ax.plot(grid_sizeL, sum_dynaL, 'c-.', label='Dyna-Q, EpiSteps' )
#ax.plot(grid_sizeL, sum_psweepL, 'r-.', label='PSweep, EpiSteps' )
print('sum_dynaL =', sum_dynaL)
print('sum_psweepL =', sum_psweepL)

gridsz_dq_suttonL = [1,2,3,4,5,6,7,8]
numup_dq_suttonL = [19252,38055.8,65859.8,148700,403090,1.11099e+06,2.42642e+06,5.75852e+06]
ax.plot(gridsz_dq_suttonL, numup_dq_suttonL, 'c:', label='Dyna-Q, Sutton' )

gridsz_ps_suttonL = [1,2,3,4,5,6,7,8]
numup_ps_suttonL = [150.234,596.865,999.17,47234.3,91829.9,169847,257342,508694]
ax.plot(gridsz_ps_suttonL, numup_ps_suttonL, 'r:', label='PSweep, Sutton' )


# Zhang Results over 10 runs.
xL = [1, 2, 3, 4, 5, 6, 7]
ypsL = [  1838.5,  15129.9,  41940.9, 122584.9, 150463.3, 311075.4, 445890.5]
ydqL = [  7275. ,  37623. ,  55078.8, 158364.6, 276714. , 413386.2, 628481.4]
ax.plot(xL, ydqL, 'c-.', label='Dyna-Q, Zhang' )
ax.plot(xL, ypsL, 'r-.', label='PSweep, Zhang' )


ax.legend()
ax.set(title='Example 8.4 Maze\n'+\
             '(Epsilon=%g, Theta=%g, #Runs=%i)\n'%(EPSILON, PRIORITY_THRESHOLD, NUM_RUNS) +\
             '(%i planning steps, alpha=%g, gamma=%g)'%(PLANNING_LOOPS, ALPHA, GAMMA))
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
plt.ylabel('# Updates to Optimum')
plt.xlabel('Gridworld Size')
plt.yscale('log')
#plt.ylim(bottom=10)

#plt.xscale('log')
#plt.xlim(left=40, right=6050)

plt.grid()
plt.tight_layout()

fig.savefig("example_8_4_psweep.png")
plt.show()