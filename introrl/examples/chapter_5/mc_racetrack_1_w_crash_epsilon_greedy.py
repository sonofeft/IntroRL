
from introrl.mc_funcs.mc_fv_epsilon_greedy import mc_epsilon_greedy
from introrl.black_box_sims.racetrack_1_w_crash_sim import RaceTrack_1_w_Crash
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


RT = RaceTrack_1_w_Crash()

pi, av = mc_epsilon_greedy( RT, initial_policy='default',
                         read_pickle_file='', #'racetrack_1_w_crash_sim', 
                         save_pickle_file='racetrack_1_w_crash_sim',
                         use_list_of_start_states=True, # use list OR single start state of environment.
                         iter_all_start_actions=True, # pick random or iterate all starting actions
                         first_visit=True, 
                         do_summ_print=False, showRunningAve=False, fmt_Q='%g', fmt_R='%g',
                         show_initial_policy=False,
                         max_num_episodes=1000, min_num_episodes=10, max_abserr=0.001, gamma=0.9,
                         iteration_prints=0,
                         max_episode_steps=10000,
                         epsilon=0.1, const_epsilon=True, half_life=500,
                         N_episodes_wo_decay=0)
                          
fig, ax = plt.subplots()
RT.plot_policy( ax, pi )

plt.show()
fig.savefig("racetrack_1_w_crash_sim.png")
