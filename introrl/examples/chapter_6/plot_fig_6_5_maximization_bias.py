import matplotlib.pyplot as plt
from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
from introrl.td_funcs.dbl_qlearning_epsilon_greedy import dbl_qlearning_epsilon_greedy
from introrl.black_box_sims.maximization_bias_mdp import MaximizationBiasMDP
from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.utils.smoother import boxcar

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 1.0
NUM_EPISODES = 300
TOTAL_RUNS = 1000
Nb_choices = 10

MB = MaximizationBiasMDP(Nb_choices=Nb_choices)
learn_tracker = LearnTracker()

left_countsL     = [0 for _ in range(NUM_EPISODES)]
dbl_left_countsL = [0 for _ in range(NUM_EPISODES)]

for num_run in range(TOTAL_RUNS):
    
    learn_tracker.clear()
    policy, state_value = \
        qlearning_epsilon_greedy( MB, learn_tracker=learn_tracker,
                              initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                              use_list_of_start_states=False, # use list OR single start state of environment.
                              do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                              pcent_progress_print=0,
                              show_banner = False,
                              max_num_episodes=NUM_EPISODES, min_num_episodes=NUM_EPISODES, max_abserr=0.001, 
                              gamma=GAMMA,
                              max_episode_steps=100,
                              epsilon=EPSILON, 
                              alpha=ALPHA)


    for iepi, sarsnL in enumerate(learn_tracker.iter_episodes()):
        for (s_hash, a_desc, reward, sn_hash) in sarsnL:
            if s_hash=='A':
                if a_desc=='Left':
                    left_countsL[iepi] += 1
    
    # ------------------- do double q-learning --------------------------
    
    learn_tracker.clear()
    policy, state_value = \
        dbl_qlearning_epsilon_greedy( MB, learn_tracker=learn_tracker,
                              initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                              use_list_of_start_states=False, # use list OR single start state of environment.
                              do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
                              pcent_progress_print=0,
                              show_banner = False,
                              max_num_episodes=NUM_EPISODES, min_num_episodes=NUM_EPISODES, max_abserr=0.001, 
                              gamma=GAMMA,
                              max_episode_steps=100,
                              epsilon=EPSILON, 
                              alpha=ALPHA)


    for iepi, sarsnL in enumerate(learn_tracker.iter_episodes()):
        for (s_hash, a_desc, reward, sn_hash) in sarsnL:
            if s_hash=='A':
                if a_desc=='Left':
                    dbl_left_countsL[iepi] += 1
    
    # ------------------------------------------------------------------------------------
    if num_run % (TOTAL_RUNS/10) == 0:
        print( 100*num_run/TOTAL_RUNS,'%', end=' ' )
print()
# ---------------------------------------------------------------
fig, ax = plt.subplots()

plt.title('Maximization Bias for Q-learning & Double Q-learning\n'+\
          'Epsilon=%g, Alpha=%g, Gamma=%g\n#Episodes=%g, TotalRuns=%g #B Choices=%g'%\
          (EPSILON,    ALPHA,    GAMMA,   NUM_EPISODES,  TOTAL_RUNS, Nb_choices))

fig.subplots_adjust(top=0.8)

plt.xlabel('Episodes')
plt.ylabel('% Left Actions from "A"')

pcent_leftL = [100.0*c/TOTAL_RUNS for c in left_countsL]
half_boxcar = 3
pcent_leftL = boxcar(pcent_leftL, half_boxcar)
plt.plot(pcent_leftL,  'r-', label='Q-learning' )

# Digitized data from Sutton & Barto
ql_epL = [2.05979,4.98766,8.14076,13.3208,19.4018,27.0593,41.023,57.239,77.9593,107.914,140.571,175.931,214.894,248.677,275.028,299.352]
ql_pcL = [50.5498,66.4618,80.4304,91.4838,94.1561,93.5487,88.3257,78.1226,62.2105,42.1686,27.1068,18.3613,14.11,13.2597,12.5309,11.8021]
plt.plot(ql_epL, ql_pcL,  'r:', label='Q-learning, Sutton' )

#zql_epL = [0.65077,3.32339,6.99824,11.0072,18.691,23.368,31.72,50.4283,73.1456,105.885,141.297,191.743,242.189,294.305]
#zql_pcL = [50.6674,69.1061,82.9009,91.2325,94.9202,95.6032,92.3252,82.4912,66.9207,43.1553,26.7653,16.9314,12.5607,12.2875]
#plt.plot(zql_epL, zql_pcL,  'r--', label='Q-learning, Zhang' )


dbl_pcent_leftL = [100.0*c/TOTAL_RUNS for c in dbl_left_countsL]
half_boxcar = 3
dbl_pcent_leftL = boxcar(dbl_pcent_leftL, half_boxcar)
plt.plot(dbl_pcent_leftL,  'g-', label='Dbl Q-learning' )


dql_epL = [2.51023,5.66333,9.26687,16.2487,25.0323,36.2934,55.212,78.4098,105.436,153.408,219.623,298.676]
dql_pcL = [50.4283,50.0639,48.6063,37.5529,26.8639,19.4545,13.0168,10.466,9.00842,7.79376,7.55083,6.70056]
plt.plot(dql_epL, dql_pcL,  'g:', label='Dbl Q-learning, Sutton' )


plt.legend()

plt.ylim(bottom=0, top=100)
plt.grid()

fig.savefig("figure_6_5_maximization_bias_nb%i.png"%Nb_choices)

plt.show()


