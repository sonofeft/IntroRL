
from introrl.mc_funcs.mc_exploring_starts import mc_exploring_starts
from introrl.black_box_sims.blackjack_sim import BlackJackSimulation

BJ = BlackJackSimulation()

pi, av = mc_exploring_starts( BJ, initial_policy='default',
                              first_visit=True, 
                              read_pickle_file='blackjack_es', 
                              save_pickle_file='blackjack_es',
                              do_summ_print=True, showRunningAve=False, fmt_Q='%g', fmt_R='%g',
                              max_num_episodes=100000000, min_num_episodes=10, max_abserr=0.000001, gamma=0.9,
                              iteration_prints=0)
                              
pi.save_diagram( BJ, inp_colorD=None, save_name='blackjack_policy',
                 show_arrows=False, scale=0.5, h_over_w=0.8,
                 show_terminal_labels=False)
