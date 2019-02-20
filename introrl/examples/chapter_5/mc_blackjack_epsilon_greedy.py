
from introrl.mc_funcs.mc_fv_epsilon_greedy import mc_epsilon_greedy
from introrl.black_box_sims.blackjack_sim import BlackJackSimulation

BJ = BlackJackSimulation()

pi, av = mc_epsilon_greedy( BJ, initial_policy='default',
                         read_pickle_file='blackjack_epsgreedy', 
                         save_pickle_file='blackjack_epsgreedy',
                         first_visit=True, 
                         do_summ_print=True, showRunningAve=False, fmt_Q='%g', fmt_R='%g',
                         show_initial_policy=True,
                         max_num_episodes=5000, min_num_episodes=10, max_abserr=0.0001, gamma=0.9,
                         iteration_prints=0,
                         max_episode_steps=10000,
                         epsilon=0.1, const_epsilon=True)
                          
