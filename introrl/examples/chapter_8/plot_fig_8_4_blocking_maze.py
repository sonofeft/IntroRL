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
PLAN_LOOPS = 10
QPLUS_FACTOR=1.0E-4

NUM_RUNS = 20

q_raveL = []
qp_raveL = []

maze_q = BlockingMaze()


for main_loop in range(NUM_RUNS):
    print('%i of %i Runs'%(1+main_loop, NUM_RUNS))


    learn_tracker_q = LearnTracker()
    learn_tracker_qp = LearnTracker()

    agent_q = DynaQAgent( environment=maze_q, 
                     learn_tracker=learn_tracker_q,
                     max_episode_steps=3000,
                     show_banner = False, do_summ_print=False, show_last_change=False, 
                     epsilon=EPSILON,
                     gamma=GAMMA,
                     alpha=ALPHA)

    agent_qp = DynaQPlusAgent( environment=maze_q, 
                     learn_tracker=learn_tracker_qp,
                     max_episode_steps=3000,
                     show_banner = False, do_summ_print=False, show_last_change=False, 
                     epsilon=EPSILON,
                     gamma=GAMMA,
                     alpha=ALPHA, qplus_factor=QPLUS_FACTOR)

    # set gates at time = 0
    maze_q.open_gate_R()    
    maze_q.close_gate_L()

    # DynaQ episodes 
    while agent_q.model.total_action_calls < 3000:
        if agent_q.model.total_action_calls >= 1000:
            maze_q.open_gate_L()
            maze_q.close_gate_R()
                
        agent_q.run_episode( 'Start', Nplanning_loops=PLAN_LOOPS)

    # set gates at time = 0
    maze_q.open_gate_R()    
    maze_q.close_gate_L()

    # DynaQ+ episodes 
    while agent_qp.model.total_action_calls < 3000:
        if agent_qp.model.total_action_calls >= 1000:
            maze_q.open_gate_L()
            maze_q.close_gate_R()
                
        agent_qp.run_episode( 'Start', Nplanning_loops=PLAN_LOOPS)

    cum_rew_qL = learn_tracker_q.cum_reward_per_step()
    cum_rew_qpL = learn_tracker_qp.cum_reward_per_step()
    
    while len(q_raveL) < min(3000,len(cum_rew_qL)):
        q_raveL.append( RunningAve() )
    for i,r in enumerate(cum_rew_qL):
        if i<3000:
            q_raveL[i].add_val(r)
    
    while len(qp_raveL) < min(3000,len(cum_rew_qpL)):
        qp_raveL.append( RunningAve() )
    for i,r in enumerate(cum_rew_qpL):
        if i<3000:
            qp_raveL[i].add_val(r)
            
#agent_q.model.summ_print(long=True)
#sys.exit()
agent_q.action_value_coll.summ_print( fmt_Q='%.3f', none_str='*', show_states=True, 
                                      show_last_change=True, show_policy=True)


fig, ax = plt.subplots()


cum_rew_qL = [R.get_ave() for R in q_raveL]
ax.plot(cum_rew_qL, 'c', label='Dyna-Q, IntroRL', linewidth=3 )

cum_rew_qpL = [R.get_ave() for R in qp_raveL]
ax.plot(cum_rew_qpL, 'r', label='Dyna-Q+, IntroRL' )

# Digitized Sutton & Barto values
q_stepL = [-5.03178,134.491,157.745,230.09,325.689,478.131,658.994,826.938,1002.63,1470.29,1483.21,1617.57,1718.33,1989.63,2356.52,2676.91,2981.79]
q_cumrL = [0.394256,0.191373,1.81443,1.81443,5.06055,11.5528,21.2911,32.0439,45.0284,44.8255,46.2457,46.4486,48.2745,54.7668,68.3599,82.3588,97.1692]
ax.plot(q_stepL, q_cumrL, 'c:', label='Dyna-Q, Sutton' )

qp_stepL = [5.30325,54.3946,67.3134,134.491,250.76,426.456,591.816,783.014,922.537,989.715,1447.04,1459.96,1568.48,1821.68,2067.14,2310.01,2565.81,2751.84,2979.21]
qp_cumrL = [-0.214391,0.800021,2.01732,2.01732,8.50955,23.9286,39.5506,57.2013,69.7801,76.6781,76.4752,77.8954,78.3011,86.4164,100.415,114.211,128.413,139.369,150.527]
ax.plot(qp_stepL, qp_cumrL, 'r:', label='Dyna-Q+, Sutton' )

# Zhang DynaQ curve
q_stepZhangL = [-5.1572,155.382,282.476,469.772,657.069,830.986,1004.9,1252.4,1489.87,1891.22,2356.11,2700.6,2991.58]
q_cumrZhangL = [0.270857,0.681621,2.53006,10.3346,21.4252,32.7212,44.6333,45.0441,46.6871,52.6432,61.2692,68.6629,75.4405]
ax.plot(q_stepZhangL, q_cumrZhangL, 'c--', label='Dyna-Q, Zhang' )

# Zhang DynaQ+ curve
qp_stepZhangL = [1.53195,115.247,249.03,369.435,556.731,764.095,998.215,1195.54,1376.15,1603.58,1907.94,2252.43,2556.79,2978.2]
qp_cumrZhangL = [-0.550669,1.09238,3.55696,10.7453,26.1489,43.401,63.7338,63.9392,64.9661,69.6899,80.5751,98.0325,113.436,136.028]
ax.plot(qp_stepZhangL, qp_cumrZhangL, 'r--', label='Dyna-Q+, Zhang' )


ax.legend()
ax.set(title='Figure 8.4 Blocking Maze\n'+\
             '(Epsilon=%g, Kappa=%g, #Runs=%i)\n'%(EPSILON, QPLUS_FACTOR, NUM_RUNS) +\
             '(%i planning steps, alpha=%g, gamma=%g)'%(PLAN_LOOPS, ALPHA, GAMMA))
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
plt.ylabel('Cumulative Reward')
plt.xlabel('Time Steps')
#plt.ylim(0, 800)
#plt.xlim(0, 3000)
fig.savefig("fig_8_4_blocking_maze.png")
plt.show()