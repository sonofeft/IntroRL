
from introrl.mc_funcs.mc_fv_epsilon_greedy import mc_epsilon_greedy
from introrl.black_box_sims.racetrack_2_sim import RaceTrack_2
from introrl.policy import Policy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


RT = RaceTrack_2()
if 1:
    pi, av = mc_epsilon_greedy( RT, initial_policy='default',
                             first_visit=True, 
                             do_summ_print=False, showRunningAve=False, fmt_Q='%g', fmt_R='%g',
                             show_initial_policy=False,
                             max_num_episodes=1000, min_num_episodes=10, max_abserr=0.001, gamma=0.9,
                             iteration_prints=0,
                             max_episode_steps=10000,
                             epsilon=0.1, const_epsilon=True, half_life=500,
                             N_episodes_wo_decay=0)
                              
    pi.save_to_pickle_file( 'racetrack_2_sim' )
else:
    pi = Policy( environment=RT )
    pi.init_from_pickle_file( 'racetrack_2_sim' )
    

fig, ax = plt.subplots()
RT.plot_policy( ax, pi )

plt.show()
fig.savefig("racetrack_2_sim.png")
