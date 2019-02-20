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
import time

from introrl.black_box_sims.cliff_walking import CliffWalkingSimulation
from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
from introrl.td_funcs.td0_epsilon_greedy import td0_epsilon_greedy
from introrl.td_funcs.expected_sarsa_eps_greedy import expected_sarsa_eps_greedy
from introrl.utils.running_ave import RunningAve
from introrl.agent_supt.learning_tracker import LearnTracker

pickle_fname = 'fig_6_3_data.pickle'
RUN_COUNT = 500  # Sutton & Barto call for 50000 runs.

ALPHA_LIST = [float(i)/100.0 for i in range(10,101,5)]
EPSILON=0.1

CW = CliffWalkingSimulation()
CW.layout.s_hash_print( none_str='*' )

learn_tracker = LearnTracker()

def read_pickle_file():
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
    return dataD
dataD = read_pickle_file()
        
# -----------------------------------------------------------------------------    
def get_expected_sarsa_data():
    if 'ExpSarsa_raveD' in dataD:
        ExpSarsa_raveD = dataD['ExpSarsa_raveD']
        ave_run_time = dataD['ExpSarsa_ave_run_time']
    else:
        ExpSarsa_raveD = {}
        ave_run_time = RunningAve()
        for alpha in ALPHA_LIST:
            ExpSarsa_raveD[alpha] = [RunningAve(), RunningAve()]

    Nruns = ExpSarsa_raveD[0.1][0].num_val
    print(Nruns,' of ExpSarsa_raveD found')

    for loop in range(Nruns, RUN_COUNT):
        for alpha in ALPHA_LIST:
        
            start_time = time.time()
            learn_tracker.clear()
            policy, state_value = \
                expected_sarsa_eps_greedy( CW,  learn_tracker=learn_tracker,
                                      initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                      use_list_of_start_states=False, # use list OR single start state of environment.
                                      do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                                      pcent_progress_print=0,
                                      show_banner = False,
                                      max_num_episodes=1000, min_num_episodes=1000, max_abserr=0.000001, 
                                      gamma=1.0,
                                      max_episode_steps=10000,
                                      epsilon=EPSILON, 
                                      alpha=alpha)

            reward_sum_per_episodeL =  learn_tracker.reward_sum_per_episode()

            ave_run_time.add_val( time.time() - start_time ) # compute average run time
            ExpSarsa_raveD[alpha][0].add_val( sum(reward_sum_per_episodeL[:100])/100.0 )
            ExpSarsa_raveD[alpha][1].add_val( sum(reward_sum_per_episodeL)/1000.0 )
                
        print('.',end='')
    print('ExpSarsa_ave_run_time = ', ave_run_time.get_ave() )
            
    dataD['ExpSarsa_raveD'] = ExpSarsa_raveD
    dataD['ExpSarsa_ave_run_time'] = ave_run_time
    save_to_pickle('ExpSarsa_raveD', 'ExpSarsa_ave_run_time')
# -----------------------------------------------------------------------------    
def get__sarsa_data():
    if 'Sarsa_raveD' in dataD:
        Sarsa_raveD = dataD['Sarsa_raveD']
        ave_run_time = dataD['Sarsa_ave_run_time']
    else:
        Sarsa_raveD = {}
        ave_run_time = RunningAve()
        for alpha in ALPHA_LIST:
            Sarsa_raveD[alpha] = [RunningAve(), RunningAve()]

    Nruns = Sarsa_raveD[0.1][0].num_val
    print(Nruns,' of Sarsa_raveD found')
        
    for loop in range(Nruns, RUN_COUNT):
        for alpha in ALPHA_LIST:
        
            start_time = time.time()
            learn_tracker.clear()
            policy, state_value = \
                sarsa_epsilon_greedy( CW,  learn_tracker=learn_tracker,
                                      initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                      use_list_of_start_states=False, # use list OR single start state of environment.
                                      do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                                      pcent_progress_print=0,
                                      show_banner = False,
                                      max_num_episodes=1000, min_num_episodes=1000, max_abserr=0.000001, 
                                      gamma=1.0,
                                      max_episode_steps=10000,
                                      epsilon=EPSILON, 
                                      alpha=alpha)
            reward_sum_per_episodeL =  learn_tracker.reward_sum_per_episode()

            ave_run_time.add_val( time.time() - start_time ) # compute average run time
            Sarsa_raveD[alpha][0].add_val( sum(reward_sum_per_episodeL[:100])/100.0 )
            Sarsa_raveD[alpha][1].add_val( sum(reward_sum_per_episodeL)/1000.0 )
                
            
        print('.',end='')
    print('Sarsa_ave_run_time = ', ave_run_time.get_ave() )
            
    dataD['Sarsa_raveD'] = Sarsa_raveD
    dataD['Sarsa_ave_run_time'] = ave_run_time
    save_to_pickle('Sarsa_raveD', 'Sarsa_ave_run_time')
# -----------------------------------------------------------------------------    
def get_qlearning_data():
    if 'Qlearn_raveD' in dataD:
        Qlearn_raveD = dataD['Qlearn_raveD']
        ave_run_time = dataD['Qlearn_ave_run_time']
    else:
        Qlearn_raveD = {}
        ave_run_time = RunningAve()
        for alpha in ALPHA_LIST:
            Qlearn_raveD[alpha] = [RunningAve(), RunningAve()]

    Nruns = Qlearn_raveD[0.1][0].num_val
    print(Nruns,' of Qlearn_raveD found')
        
    for loop in range(Nruns, RUN_COUNT):
        for alpha in ALPHA_LIST:
        
            start_time = time.time()
            learn_tracker.clear()
            policy, state_value = \
                qlearning_epsilon_greedy( CW,  learn_tracker=learn_tracker,
                                      initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                      use_list_of_start_states=False, # use list OR single start state of environment.
                                      do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                                      pcent_progress_print=0,
                                      show_banner = False,
                                      max_num_episodes=1000, min_num_episodes=1000, max_abserr=0.000001, 
                                      gamma=1.0,
                                      max_episode_steps=10000,
                                      epsilon=EPSILON, 
                                      alpha=alpha)
            reward_sum_per_episodeL =  learn_tracker.reward_sum_per_episode()

            ave_run_time.add_val( time.time() - start_time ) # compute average run time
            Qlearn_raveD[alpha][0].add_val( sum(reward_sum_per_episodeL[:100])/100.0 )
            Qlearn_raveD[alpha][1].add_val( sum(reward_sum_per_episodeL)/1000.0 )
                
            
        print('.',end='')
    print('Qlearn_ave_run_time = ', ave_run_time.get_ave() )
            
    dataD['Qlearn_raveD'] = Qlearn_raveD
    dataD['Qlearn_ave_run_time'] = ave_run_time
    save_to_pickle('Qlearn_raveD', 'Qlearn_ave_run_time')

# -----------------------------------------------------------------------------    
def save_to_pickle( *argv ):
    print('---------------- Saving Data ---------------------')
    
    # get any changes to pickle file and add new data to it.
    current_dataD = read_pickle_file()
    for arg in argv:
        if arg in dataD:        
            current_dataD[arg] = dataD[arg]
    
    fileObject = open(pickle_fname,'wb')
    pickle.dump(current_dataD, fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
    fileObject.close()
    print('Saved dataD to file:',pickle_fname)

if __name__ == "__main__": # pragma: no cover
    
    get_expected_sarsa_data()
    get__sarsa_data()
    get_qlearning_data()
    
