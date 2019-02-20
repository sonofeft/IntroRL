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

from introrl.black_box_sims.blocking_maze import BlockingMaze
from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.policy import Policy
from introrl.utils.running_ave import RunningAve
from introrl.agents.dyna_q_agent import DynaQAgent
from introrl.agents.dyna_qplus_agent import DynaQPlusAgent

ALPHA = 1.0
GAMMA = 0.95
EPSILON=0.1
PLAN_LOOPS = 50
QPLUS_FACTOR=1.0E-3

NUM_RUNS = 10

q_raveL = []
qp_raveL = []

maze_q = BlockingMaze()


for main_loop in range(NUM_RUNS):
    print('%i of %i Runs'%(1+main_loop, NUM_RUNS))


    learn_tracker_q = LearnTracker()
    learn_tracker_qp = LearnTracker()

    agent_q = DynaQAgent( environment=maze_q, 
                     learn_tracker=learn_tracker_q,
                     max_episode_steps=6000,
                     show_banner = False, do_summ_print=False, show_last_change=False, 
                     epsilon=EPSILON,
                     gamma=GAMMA,
                     alpha=ALPHA)

    agent_qp = DynaQPlusAgent( environment=maze_q, 
                     learn_tracker=learn_tracker_qp,
                     max_episode_steps=6000,
                     show_banner = False, do_summ_print=False, show_last_change=False, 
                     epsilon=EPSILON,
                     gamma=GAMMA,
                     alpha=ALPHA, qplus_factor=QPLUS_FACTOR)

    # set gates at time = 0
    maze_q.open_gate_L()    
    maze_q.close_gate_R()

    # DynaQ episodes 
    while agent_q.model.total_action_calls < 6000:
        if agent_q.model.total_action_calls >= 3000:
            maze_q.open_gate_R()
                
        agent_q.run_episode( 'Start', Nplanning_loops=PLAN_LOOPS)

    # set gates at time = 0
    maze_q.open_gate_L()    
    maze_q.close_gate_R()

    # DynaQ+ episodes 
    while agent_qp.model.total_action_calls < 6000:
        if agent_qp.model.total_action_calls >= 3000:
            maze_q.open_gate_R()
                
        agent_qp.run_episode( 'Start', Nplanning_loops=PLAN_LOOPS)

    cum_rew_qL = learn_tracker_q.cum_reward_per_step()
    cum_rew_qpL = learn_tracker_qp.cum_reward_per_step()
    
    while len(q_raveL) < min(6000,len(cum_rew_qL)):
        q_raveL.append( RunningAve() )
    for i,r in enumerate(cum_rew_qL):
        if i<6000:
            q_raveL[i].add_val(r)
    
    while len(qp_raveL) < min(6000,len(cum_rew_qpL)):
        qp_raveL.append( RunningAve() )
    for i,r in enumerate(cum_rew_qpL):
        if i<6000:
            qp_raveL[i].add_val(r)
            
#agent_q.model.summ_print(long=True)
#sys.exit()
#agent_q.action_value_coll.summ_print( fmt_Q='%.3f', none_str='*', show_states=True, 
#                                      show_last_change=True, show_policy=True)


fig, ax = plt.subplots()


cum_rew_qL = [R.get_ave() for R in q_raveL]
ax.plot(cum_rew_qL, 'c', label='Dyna-Q, IntroRL', linewidth=3 )

cum_rew_qpL = [R.get_ave() for R in qp_raveL]
ax.plot(cum_rew_qpL, 'r', label='Dyna-Q+, IntroRL' )

# Digitized Sutton & Barto
tstep_qL = [6.11195,310.532,532.807,924.203,1325.26,1793.97,2354.49,2852.2,3461.04,3944.24,4528.92,5200.58,5925.39]
cumr_qL  = [0.464093,0.464093,3.99981,17.6376,36.8315,61.5815,92.8979,122.699,159.571,187.352,221.699,263.623,308.072]
ax.plot(tstep_qL, cumr_qL, 'c:', label='Dyna-Q, Sutton' )

tstep_qpL = [-3.55217,305.7,537.639,919.371,1426.74,1938.94,2446.3,3098.63,3552.84,3920.08,4470.94,5060.45,5959.21]
cumr_qpL = [-0.0410098,2.4845,10.5661,30.2651,59.056,86.3315,113.102,149.469,177.755,204.526,255.541,313.123,400.505]
ax.plot(tstep_qpL, cumr_qpL, 'r:', label='Dyna-Q+, Sutton' )


ax.legend()
ax.set(title='Figure 8.t Shortcut Maze\n'+\
             '(Epsilon=%g, Kappa=%g, #Runs=%i)\n'%(EPSILON, QPLUS_FACTOR, NUM_RUNS) +\
             '(%i planning steps, alpha=%g, gamma=%g)'%(PLAN_LOOPS, ALPHA, GAMMA))
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
plt.ylabel('Cumulative Reward')
plt.xlabel('Time Steps')
#plt.ylim(0, 800)
#plt.xlim(0, 6000)
fig.savefig("fig_8_5_shortcut_maze.png")
plt.show()