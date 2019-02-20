import matplotlib.pyplot as plt

from introrl.black_box_sims.cliff_walking import CliffWalkingSimulation
from introrl.td_funcs.qlearning_epsilon_greedy import qlearning_epsilon_greedy
from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
from introrl.td_funcs.td0_epsilon_greedy import td0_epsilon_greedy

from introrl.utils.running_ave import RunningAve
from introrl.agent_supt.learning_tracker import LearnTracker

CW = CliffWalkingSimulation()
CW.layout.s_hash_print( none_str='*' )

Sarsa_raveL = []
Qlearn_raveL = []
TD0_raveL = []

RUN_COUNT = 1000
ALPHA=0.5
EPSILON=0.1

learn_tracker = LearnTracker()

for loop in range(RUN_COUNT):
    
    learn_tracker.clear()
    policy_t, state_value_t = \
        td0_epsilon_greedy( CW,  learn_tracker=learn_tracker,
                              initial_Vs=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                              use_list_of_start_states=False, # use list OR single start state of environment.
                              do_summ_print=False, show_last_change=False, fmt_V='%g', fmt_R='%g',
                              max_num_episodes=500, min_num_episodes=10, max_abserr=0.001, 
                              gamma=1.0,
                              max_episode_steps=1000,
                              epsilon=EPSILON, 
                              alpha=ALPHA)

    reward_sum_per_episodeL_t = learn_tracker.reward_sum_per_episode()
    
    while len(reward_sum_per_episodeL_t) > len(TD0_raveL):
        TD0_raveL.append( RunningAve() )
    for R,r in zip(TD0_raveL,  reward_sum_per_episodeL_t):
        R.add_val( r )
        
    
    if 1:
        learn_tracker.clear()
        policy_s, state_value_s = \
            sarsa_epsilon_greedy( CW,  learn_tracker=learn_tracker,
                                  initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                                  use_list_of_start_states=False, # use list OR single start state of environment.
                                  do_summ_print=False, show_last_change=False, fmt_Q='%g', fmt_R='%g',
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

# make a list of the averages
reward_sum_per_episodeL_q = [R.get_ave() for R in Qlearn_raveL]
reward_sum_per_episodeL_s = [R.get_ave() for R in Sarsa_raveL]
reward_sum_per_episodeL_t = [R.get_ave() for R in TD0_raveL]


fig, ax = plt.subplots()
plt.title('TD(0) Vs. Sarsa Vs. Q-Learning Cliff Walking\nEpsilon=%g, Alpha=%g\n(averaged over %i runs)'%(EPSILON, ALPHA, RUN_COUNT) )

plt.xlabel('Episodes')
plt.ylabel('Reward Sum for Episode')
plt.plot( reward_sum_per_episodeL_s, 'c', label='Sarsa' )
plt.plot( reward_sum_per_episodeL_t, 'g', label='TD(0)' )
plt.plot( reward_sum_per_episodeL_q, 'r', label='Q-learning' )

plt.legend()

plt.ylim(bottom=-100)
plt.grid()
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')

fig.savefig("example_6_6_cliff_walking_w_td0.png")
    
#print(steps_per_episodeL)
#print( min(steps_per_episodeL) )
plt.show()
