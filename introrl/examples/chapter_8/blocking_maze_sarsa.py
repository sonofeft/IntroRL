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

from introrl.black_box_sims.blocking_maze import BlockingMaze
from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.policy import Policy
from introrl.utils.running_ave import RunningAve
from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy

ALPHA = 0.5
GAMMA = 0.95
EPSILON=0.25

NUM_RUNS = 20

q_raveL = []
qp_raveL = []

for main_loop in range(NUM_RUNS):
    print('\n%i of %i Runs'%(1+main_loop, NUM_RUNS))

    maze_q = BlockingMaze()

    learn_tracker_q = LearnTracker()
    
    # set gates at time = 0
    maze_q.open_gate_R()    
    maze_q.close_gate_L()

    # episodes 
    time_stamp = 0
    read_pickle_file=''
    for i in range(400):
        if time_stamp >= 1000:
            maze_q.open_gate_L()
            maze_q.close_gate_R()
            
        if time_stamp >= 3000:
            break
        print(i,end=' ')
        
        policy, action_value = \
            sarsa_epsilon_greedy( maze_q, learn_tracker=learn_tracker_q,
                                  initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                  read_pickle_file=read_pickle_file,
                                  save_pickle_file='blocking_sarsa',
                                  use_list_of_start_states=False, # use list OR single start state of environment.
                                  do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                                  show_banner = False,
                                  max_num_episodes=1, min_num_episodes=11, max_abserr=0.001, gamma=GAMMA,
                                  iteration_prints=0,
                                  max_episode_steps=3000,
                                  epsilon=EPSILON,
                                  alpha=ALPHA)

        time_stamp = sum( learn_tracker_q.steps_per_episode() )
        read_pickle_file='blocking_sarsa'
        
    cum_rew_qL = learn_tracker_q.cum_reward_per_step()
    
    while len(q_raveL) < len(cum_rew_qL):
        q_raveL.append( RunningAve() )
    for i,r in enumerate(cum_rew_qL):
        q_raveL[i].add_val(r)
        

fig, ax = plt.subplots()


cum_rew_qL = [R.get_ave() for R in q_raveL]
ax.plot(cum_rew_qL, 'c', label='SARSA', linewidth=3 )

# Digitized Sutton & Barto values
q_stepL = [-5.03178,134.491,157.745,230.09,325.689,478.131,658.994,826.938,1002.63,1470.29,1483.21,1617.57,1718.33,1989.63,2356.52,2676.91,2981.79]
q_cumrL = [0.394256,0.191373,1.81443,1.81443,5.06055,11.5528,21.2911,32.0439,45.0284,44.8255,46.2457,46.4486,48.2745,54.7668,68.3599,82.3588,97.1692]
ax.plot(q_stepL, q_cumrL, 'c:', label='Dyna-Q, Sutton' )

qp_stepL = [5.30325,54.3946,67.3134,134.491,250.76,426.456,591.816,783.014,922.537,989.715,1447.04,1459.96,1568.48,1821.68,2067.14,2310.01,2565.81,2751.84,2979.21]
qp_cumrL = [-0.214391,0.800021,2.01732,2.01732,8.50955,23.9286,39.5506,57.2013,69.7801,76.6781,76.4752,77.8954,78.3011,86.4164,100.415,114.211,128.413,139.369,150.527]
ax.plot(qp_stepL, qp_cumrL, 'r:', label='Dyna-Q+, Sutton' )


ax.legend()
ax.set(title='Modified Figure 8.4 Blocking Maze, SARSA\n'+\
             '(Epsilon=%g, #Runs=%i)\n'%(EPSILON, NUM_RUNS) +\
             '( alpha=%g, gamma=%g)'%( ALPHA, GAMMA))
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
plt.ylabel('Cumulative Reward')
plt.xlabel('Time Steps')
#plt.ylim(0, 800)
#plt.xlim(0, 3000)
fig.savefig("blocking_maze_sarsa.png")
plt.show()