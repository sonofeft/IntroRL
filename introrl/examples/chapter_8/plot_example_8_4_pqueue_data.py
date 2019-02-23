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

grid_sizeL = [1, 2, 3, 4, 5, 6, 7, 8]

# number of updates
numup_dynaL = [53034, 268164, 657378, 1541982, 2555574, 3498438, 5773146, 8804724]
numup_psweepL = [6560, 57760, 180785, 436415, 840685, 1561835, 2524119, 4803181]

# forgot to divide by # runs
numup_dynaL = [n/NUM_RUNS for n in numup_dynaL]
numup_psweepL = [n/NUM_RUNS for n in numup_psweepL]

# sum of episode steps
sum_dynaL = [883.9, 4469.4, 10956.3, 25699.7, 42592.9, 58307.3, 96219.1, 146745.4]
sum_psweepL = [824.9, 3591.2, 9840.6, 24654.2, 32764.8, 61414.1, 93895.3, 139546.8]

fig, ax = plt.subplots()


ax.plot(grid_sizeL, numup_dynaL, 'c-', label='Dyna-Q, IntroRL' )
ax.plot(grid_sizeL, numup_psweepL, 'r-', label='PSweep, IntroRL' )

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

fig.savefig("example_8_4_psweep_data.png")
plt.show()