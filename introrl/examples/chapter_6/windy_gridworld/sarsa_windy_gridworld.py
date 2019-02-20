import matplotlib.pyplot as plt

from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
from introrl.agent_supt.episode_maker import make_episode
from introrl.agent_supt.episode_summ_print import epi_summ_print

from introrl.mdp_data.windy_gridworld import get_gridworld
from introrl.agent_supt.learning_tracker import LearnTracker

gridworld = get_gridworld( step_reward=-1 )
learn_tracker = LearnTracker()

policy, state_value =\
    sarsa_epsilon_greedy( gridworld, learn_tracker=learn_tracker,
                          initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                          read_pickle_file='', 
                          save_pickle_file='',
                          use_list_of_start_states=False, # use list OR single start state of environment.
                          do_summ_print=True, show_last_change=True, fmt_Q='%g', fmt_R='%g',
                          max_num_episodes=170, min_num_episodes=10, max_abserr=0.001, gamma=1.0,
                          iteration_prints=0,
                          max_episode_steps=10000,
                          epsilon=0.1, const_epsilon=True,
                          alpha=0.5, const_alpha=True)

print('_'*55)
score = gridworld.get_policy_score( policy, start_state_hash=None, step_limit=1000)
print('Policy Score =', score, ' = (r_sum, n_steps, msg)')

steps_per_episodeL = learn_tracker.steps_per_episode()

print( gridworld.get_info() )

episode = make_episode( gridworld.start_state_hash, policy, gridworld, 
                        gridworld.terminal_set, max_steps=20 )

epi_summ_print(episode, policy, gridworld, show_rewards=False,
               show_env_states=True, none_str='*')


fig, ax = plt.subplots()
plt.title('SARSA Windy Gridworld')
if 1:
    plt.xlabel('Time Steps')
    plt.ylabel('Episodes')
    cum_stepsL = [0]
    for steps in steps_per_episodeL:
        cum_stepsL.append( cum_stepsL[-1] + steps )
    plt.plot( cum_stepsL, list(range(len(cum_stepsL))), label="IntroRL" )

    # Example 6.5, Digitized Sutton & Barto Data
    time_stepL = [1,1567.48,2112.46,2492.59,2734.8,2896.28,3094.75,3188.95,3458.07,3754.1,4070.32,4410.09,4420.18,4551.38,4655.66,4793.59,4918.06,5163.63,5315.01,5419.05,5618.51,5793.54,5952.29,6070.33,6163.95,6347.12,6538.43,6790.8,6986.19,7205.99,7405.45,7535.7,7696.95,7828.06]
    episodesL = [0.205302,0.205302,0.968965,5.29639,7.07827,10.7693,11.7875,14.0785,15.3513,18.4059,19.6787,25.4062,27.8244,28.7154,31.5155,35.0792,38.0066,47.0433,51.8798,56.4569,62.4631,68.3153,74.7835,79.2497,84.3319,93.2642,101.58,113.285,123.449,133.306,143.316,151.786,160.717,168.834]
    plt.plot( time_stepL, episodesL, label='Sutton' )
    
    # Digitized from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb
    t_stepL = [0.,971.925,1642.18,2476.95,2927.85,3360.47,3652.94,4000.25,4554.74,5090.94,5700.26,6266.93,6833.6,7491.67,8021.78,8612.82]
    episodL = [0.,1.55979,4.25581,11.8046,20.4319,23.1279,29.5983,33.1031,45.5048,57.3672,73.0041,90.2585,113.983,141.752,167.095,198.908]
    plt.plot( t_stepL, episodL, label='Britz' )
    
    
    plt.legend()
    plt.grid()

    fig.savefig("example_6_5_windy_gridworld.png")
else:
    plt.ylabel('Steps per Episode')
    plt.xlabel('Episodes')
    plt.semilogy(list(range(1, len(steps_per_episodeL)+1)), steps_per_episodeL )

    fig.savefig("example_6_5_windy_gridworld_v2.png")
    
#print(steps_per_episodeL)
#print( min(steps_per_episodeL) )
plt.show()
