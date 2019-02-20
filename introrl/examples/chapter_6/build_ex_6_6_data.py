#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
import os
import random
import pickle

from introrl.black_box_sims.cliff_walking import CliffWalkingSimulation
from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
from introrl.td_funcs.td0_epsilon_greedy import td0_epsilon_greedy
from introrl.td_funcs.expected_sarsa_eps_greedy import expected_sarsa_eps_greedy
from introrl.utils.running_ave import RunningAve
from introrl.agent_supt.learning_tracker import LearnTracker

pickle_fname = 'ex_6_6_data.pickle'
RUN_COUNT = 100
ALPHA=0.5
EPSILON=0.1

CW = CliffWalkingSimulation()
CW.layout.s_hash_print( none_str='*' )

learn_tracker = LearnTracker()

if not os.path.isfile( pickle_fname ):
    print('Pickle File NOT found:', pickle_fname)
    print('------ Will recreate all data and save to: %s-------'%pickle_fname)
    dataD = {} # index=method, value=data list
else:
    fileObject = open(pickle_fname,'rb')
    dataD = pickle.load(fileObject)

    fileObject.close()
    print('Read Data from file:',pickle_fname)
    print( dataD.keys() )
# -----------------------------------------------------------------------------    
def get_expected_sarsa_data():
    if 'ExpSarsa_raveL' in dataD:
        ExpSarsa_raveL = dataD['ExpSarsa_raveL']
        Nruns = ExpSarsa_raveL[0].num_val
        print(Nruns,' of ExpSarsa_raveL found')
    else:
        ExpSarsa_raveL = []
        Nruns = 0
        
    for loop in range(Nruns, RUN_COUNT):
        
        learn_tracker.clear()
        policy, state_value = \
            expected_sarsa_eps_greedy( CW,   learn_tracker=learn_tracker,
                                  initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                  use_list_of_start_states=False, # use list OR single start state of environment.
                                  do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                                  pcent_progress_print=0,
                                  show_banner = False,
                                  max_num_episodes=500, min_num_episodes=10, max_abserr=0.001, 
                                  gamma=1.0,
                                  max_episode_steps=1000,
                                  epsilon=EPSILON, 
                                  alpha=ALPHA)
        reward_sum_per_episodeL =  learn_tracker.reward_sum_per_episode()

        while len(reward_sum_per_episodeL) > len(ExpSarsa_raveL):
            ExpSarsa_raveL.append( RunningAve() )
        for R,r in zip(ExpSarsa_raveL,  reward_sum_per_episodeL):
            R.add_val( r )
    dataD['ExpSarsa_raveL'] = ExpSarsa_raveL
    save_to_pickle()
# -----------------------------------------------------------------------------    
def get_td0_data():
    if 'TD0_raveL' in dataD:
        TD0_raveL = dataD['TD0_raveL']
        Nruns = TD0_raveL[0].num_val
        print(Nruns,' of TD0_raveL found')
    else:
        TD0_raveL = []
        Nruns = 0
        
    for loop in range(Nruns, RUN_COUNT):
        
        learn_tracker.clear()
        policy, state_value = \
            td0_epsilon_greedy( CW,   learn_tracker=learn_tracker,
                                  initial_Vs=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                  use_list_of_start_states=False, # use list OR single start state of environment.
                                  do_summ_print=False, show_last_change=False, fmt_V='%g', fmt_R='%g',
                                  pcent_progress_print=0,
                                  show_banner = False,
                                  max_num_episodes=500, min_num_episodes=10, max_abserr=0.001, 
                                  gamma=1.0,
                                  max_episode_steps=1000,
                                  epsilon=EPSILON, 
                                  alpha=ALPHA)
        reward_sum_per_episodeL =  learn_tracker.reward_sum_per_episode()

        while len(reward_sum_per_episodeL) > len(TD0_raveL):
            TD0_raveL.append( RunningAve() )
        for R,r in zip(TD0_raveL,  reward_sum_per_episodeL):
            R.add_val( r )
    dataD['TD0_raveL'] = TD0_raveL
    save_to_pickle()
# -----------------------------------------------------------------------------    
def get__sarsa_data():
    if 'Sarsa_raveL' in dataD:
        Sarsa_raveL = dataD['Sarsa_raveL']
        Nruns = Sarsa_raveL[0].num_val
        print(Nruns,' of Sarsa_raveL found')
    else:
        Sarsa_raveL = []
        Nruns = 0
        
    for loop in range(Nruns, RUN_COUNT):
        
        learn_tracker.clear()
        policy, state_value = \
            sarsa_epsilon_greedy( CW,   learn_tracker=learn_tracker,
                                  initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                  use_list_of_start_states=False, # use list OR single start state of environment.
                                  do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                                  pcent_progress_print=0,
                                  show_banner = False,
                                  max_num_episodes=500, min_num_episodes=10, max_abserr=0.001, 
                                  gamma=1.0,
                                  max_episode_steps=1000,
                                  epsilon=EPSILON, 
                                  alpha=ALPHA)
        reward_sum_per_episodeL =  learn_tracker.reward_sum_per_episode()

        while len(reward_sum_per_episodeL) > len(Sarsa_raveL):
            Sarsa_raveL.append( RunningAve() )
        for R,r in zip(Sarsa_raveL,  reward_sum_per_episodeL):
            R.add_val( r )
    dataD['Sarsa_raveL'] = Sarsa_raveL
    save_to_pickle()
# -----------------------------------------------------------------------------    
def get_qlearning_data():
    if 'Qlearn_raveL' in dataD:
        Qlearn_raveL = dataD['Qlearn_raveL']
        Nruns = Qlearn_raveL[0].num_val
        print(Nruns,' of Qlearn_raveL found')
    else:
        Qlearn_raveL = []
        Nruns = 0
        
    for loop in range(Nruns, RUN_COUNT):
        
        learn_tracker.clear()
        policy, state_value = \
            qlearning_epsilon_greedy( CW,   learn_tracker=learn_tracker,
                                  initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                  use_list_of_start_states=False, # use list OR single start state of environment.
                                  do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                                  pcent_progress_print=0,
                                  show_banner = False,
                                  max_num_episodes=500, min_num_episodes=10, max_abserr=0.001, 
                                  gamma=1.0,
                                  max_episode_steps=1000,
                                  epsilon=EPSILON, 
                                  alpha=ALPHA)
        reward_sum_per_episodeL =  learn_tracker.reward_sum_per_episode()

        while len(reward_sum_per_episodeL) > len(Qlearn_raveL):
            Qlearn_raveL.append( RunningAve() )
        for R,r in zip(Qlearn_raveL,  reward_sum_per_episodeL):
            R.add_val( r )
    dataD['Qlearn_raveL'] = Qlearn_raveL
    save_to_pickle()

# -----------------------------------------------------------------------------    
def save_to_pickle():
    print('---------------- Saving Data ---------------------')
    fileObject = open(pickle_fname,'wb')
    pickle.dump(dataD, fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
    fileObject.close()
    print('Saved ActionValueColl to file:',pickle_fname)

if __name__ == "__main__": # pragma: no cover
    
    get_expected_sarsa_data()
    get_td0_data()
    get__sarsa_data()
    get_qlearning_data()
    
