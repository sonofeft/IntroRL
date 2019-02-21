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

from introrl.mdp_data.sutton_dyna_grid_xN import get_gridworld
from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.policy import Policy
from introrl.agents.dyna_q_agent import DynaQAgent

learn_tracker_0 = LearnTracker()
learn_tracker_5 = LearnTracker()
learn_tracker_50 = LearnTracker()

gridworld = get_gridworld( N_mult=3 )
#gridworld.summ_print(long=False)
print('-'*77)    

agent_0 = DynaQAgent( environment=gridworld, 
                      learn_tracker=learn_tracker_0,
                      gamma=0.95)
agent_5 = DynaQAgent( environment=gridworld, 
                      learn_tracker=learn_tracker_5,
                      gamma=0.95)
agent_50 = DynaQAgent( environment=gridworld, 
                       learn_tracker=learn_tracker_50,
                       gamma=0.95)
                  
# use same 1st episode for all agents.
agent_0.run_episode( (2,0), Nplanning_loops=0)

sarsnL = learn_tracker_0.get_episode_sarsn_list(0)
agent_5.run_episode( (2,0), Nplanning_loops=5, iter_sarsn=iter(sarsnL) )
agent_50.run_episode( (2,0), Nplanning_loops=50, iter_sarsn=iter(sarsnL))

# episodes 2 to 50
for i in range(49):
    print(i,end=' ')
    agent_0.run_episode( (2,0), Nplanning_loops=0)
    agent_5.run_episode( (2,0), Nplanning_loops=5)
    agent_50.run_episode( (2,0), Nplanning_loops=50)

fig, ax = plt.subplots()


step_0L = learn_tracker_0.steps_per_episode()[1:]
ax.plot(step_0L, 'c', label='0 planning steps' )

step_5L = learn_tracker_5.steps_per_episode()[1:]
ax.plot(step_5L, 'g', label='5 planning steps' )

step_50L = learn_tracker_50.steps_per_episode()[1:]
ax.plot(step_50L, 'r', label='50 planning steps' )

ax.legend()
ax.set(title='Figure 8.2 Dyna Maze xN\n(common 1st episode)')
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
plt.ylabel('Steps per Episode')
plt.xlabel('Episodes')
plt.ylim(0, 800)
fig.savefig("fig_8_2_dyna_maze_xN.png")
plt.show()