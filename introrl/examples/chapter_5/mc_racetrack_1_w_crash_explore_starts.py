
from introrl.mc_funcs.mc_exploring_starts import mc_exploring_starts
from introrl.black_box_sims.racetrack_1_w_crash_sim import RaceTrack_1_w_Crash

RT = RaceTrack_1_w_Crash()

if 1:
    pi, av = mc_exploring_starts( RT, initial_policy='default',
                                  first_visit=True, 
                                  do_summ_print=True, showRunningAve=False, fmt_Q='%g', fmt_R='%g',
                                  show_initial_policy=False,
                                  max_num_episodes=10000, min_num_episodes=10, max_abserr=0.001, gamma=0.9,
                                  iteration_prints=0)
    pi.save_to_pickle_file( 'racetrack_1_w_crash_es_sim' )
else:
    pi = Policy( environment=RT )
    pi.init_from_pickle_file( 'racetrack_1_w_crash_es_sim' )
    

fig, ax = plt.subplots()
RT.plot_policy( ax, pi )

plt.show()
fig.savefig("racetrack_1_w_crash_es_sim.png")
                                  