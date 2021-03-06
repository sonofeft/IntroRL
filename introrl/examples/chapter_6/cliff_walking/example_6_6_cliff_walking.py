import matplotlib.pyplot as plt
import time
from introrl.black_box_sims.cliff_walking import CliffWalkingSimulation
from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy

from introrl.utils.running_ave import RunningAve
from introrl.utils.smoother import boxcar
from introrl.agent_supt.learning_tracker import LearnTracker

CW = CliffWalkingSimulation()
CW.layout.s_hash_print( none_str='*' )

Sarsa_raveL = []
Qlearn_raveL = []

RUN_COUNT = 50
ALPHA=0.5
EPSILON=0.1

learn_tracker = LearnTracker()

start_time = time.time()
for loop in range(RUN_COUNT):
    
    learn_tracker.clear()
    policy_s, state_value_s = \
        sarsa_epsilon_greedy( CW,  learn_tracker=learn_tracker,
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
                              
    reward_sum_per_episodeL_s = learn_tracker.reward_sum_per_episode()

    while len(reward_sum_per_episodeL_s) > len(Sarsa_raveL):
        Sarsa_raveL.append( RunningAve() )
    for R,r in zip(Sarsa_raveL,  reward_sum_per_episodeL_s):
        R.add_val( r )
    
    
    learn_tracker.clear()
    policy_q, state_value_q = \
        qlearning_epsilon_greedy( CW,  learn_tracker=learn_tracker,
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
                              
    reward_sum_per_episodeL_q = learn_tracker.reward_sum_per_episode()

    while len(reward_sum_per_episodeL_q) > len(Qlearn_raveL):
        Qlearn_raveL.append( RunningAve() )
    for R,r in zip(Qlearn_raveL,  reward_sum_per_episodeL_q):
        R.add_val( r )
    
    ten_pcent = int(RUN_COUNT / 10)
    if (loop%ten_pcent==0):
        sec_per_loop = (time.time() - start_time) / float(loop+1)
        sec_to_go = (RUN_COUNT - loop) * sec_per_loop
        print('%3i/%i'%(loop,RUN_COUNT), 
              ' to go = %8.0f sec'%sec_to_go , 
              ' = %8.2f min'%(sec_to_go/60,))

# make a list of the averages
reward_sum_per_episodeL_q = [R.get_ave() for R in Qlearn_raveL]
reward_sum_per_episodeL_s = [R.get_ave() for R in Sarsa_raveL]

half_boxcar = 5
reward_sum_per_episodeL_q = boxcar(reward_sum_per_episodeL_q, half_boxcar)
reward_sum_per_episodeL_s = boxcar(reward_sum_per_episodeL_s, half_boxcar)

fig, ax = plt.subplots()
plt.title('Sarsa Vs. Q-Learning Cliff Walking\nEpsilon=%g, Alpha=%g\n(averaged over %i runs)'%(EPSILON, ALPHA, RUN_COUNT) )

plt.xlabel('Episodes')
plt.ylabel('Reward Sum for Episode')
plt.plot( reward_sum_per_episodeL_s, 'c', label='Sarsa' )
plt.plot( reward_sum_per_episodeL_q, 'r', label='Q-learning' )

# Digitized Sutton & Barto Data
q_epL = [10.1028,11.7866,15.4909,21.8893,25.2569,27.6142,34.0126,39.4008,42.0949,46.8095,48.4933,53.2079,61.9636,67.3518,69.0356,72.7399,77.1178,79.8119,84.8632,88.5676,93.9557,103.048,108.773,112.477,119.549,127.968,133.357,137.734,142.112,146.49,152.552,153.562,157.266,163.328,168.379,172.421,178.819,181.85,190.269,197.677,201.045,206.433,209.464,218.22,227.986,238.425,249.875,260.315,265.703,268.06,274.458,278.836,284.898,302.073,307.124,313.186,317.564,321.941,331.708,334.738,341.474,347.535,351.913,357.301,368.077,375.823,385.926,396.028,406.131,420.949,430.378,438.797,445.532,453.614,459.676,476.514,481.565,484.596,492.005]
q_rwdL = [-100.103,-76.7412,-61.6803,-49.1865,-46.876,-48.5875,-48.4163,-49.7855,-51.3258,-48.4163,-45.5924,-43.2819,-41.4849,-44.0521,-46.0203,-47.0472,-45.8491,-44.2232,-45.4213,-46.5337,-45.8491,-48.7586,-48.1596,-47.8173,-49.1009,-48.9298,-46.876,-45.3357,-45.3357,-44.2232,-46.1059,-47.2183,-48.4163,-48.9298,-47.7318,-41.057,-41.3993,-44.8222,-49.4432,-47.8173,-48.5019,-48.6731,-46.1914,-50.0422,-42.5973,-50.6413,-49.7855,-44.3088,-45.4213,-49.3577,-50.3845,-44.7367,-43.1963,-49.5288,-45.3357,-44.7367,-46.7049,-45.2501,-47.5606,-45.4213,-46.3626,-52.695,-53.294,-50.1278,-48.7586,-52.0104,-46.5337,-49.9567,-43.4531,-53.294,-51.5826,-46.277,-42.7685,-49.3577,-49.8711,-44.0521,-45.2501,-45.1645,-37.035]

s_epL = [11.5792,11.161,13.252,13.252,15.1339,18.6886,22.223,25.5906,28.3459,37.8364,47.633,53.4498,67.5324,75.7983,80.3905,89.8809,99.0653,108.556,116.516,122.638,128.149,137.946,148.355,155.09,170.703,173.765,182.337,192.745,208.665,216.625,232.544,237.136,249.076,274.18,286.12,292.242,299.896,307.55,322.245,331.429,351.022,362.656,366.636,379.187,388.984,399.393,405.516,417.455,427.864,433.681,449.907,454.499,463.683,475.011,491.836]
s_rwdL = [-100.256,-79.0557,-74.2735,-70.7135,-66.5159,-57.483,-45.9687,-40.8343,-38.1893,-32.4325,-28.7762,-26.1312,-29.4763,-26.1312,-26.1312,-29.0096,-25.6644,-26.2868,-27.9204,-25.5866,-24.6531,-25.9756,-22.2415,-21.6969,-24.6531,-24.6531,-26.3645,-23.4084,-26.209,-26.2868,-22.8638,-22.7082,-26.6757,-22.7082,-25.431,-24.5753,-26.5979,-25.5866,-29.1651,-25.5866,-25.8978,-29.4763,-29.0096,-24.6531,-27.1425,-23.9529,-24.8865,-29.0096,-23.3306,-22.3192,-27.9204,-26.8313,-27.3759,-25.5088,-25.4372]

plt.plot( s_epL, s_rwdL,  'c:', label='Sarsa, Sutton Pub.' )
plt.plot( q_epL, q_rwdL,  'r:', label='Q-learning, Sutton Pub.' )

plt.legend()

plt.ylim(bottom=-100, top=0.0)
plt.grid()
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
s ='%g'%ALPHA
s = s.replace('.','_')
fig.savefig("example_6_6_cliff_walking_alpha_%s.png"%s  )
    
plt.show()
