from introrl.td_funcs.sarsa_epsilon_greedy import sarsa_epsilon_greedy
from introrl.agent_supt.episode_maker import make_episode
from introrl.agent_supt.episode_summ_print import epi_summ_print

from introrl.mdp_data.windy_kings_gridworld import get_env

gridworld = get_env( step_reward=-1 )

policy, state_value = \
    sarsa_epsilon_greedy( gridworld, 
                          initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                          read_pickle_file='', 
                          save_pickle_file='',
                          use_list_of_start_states=False, # use list OR single start state of environment.
                          do_summ_print=True, show_last_change=True, fmt_Q='%g', fmt_R='%g',
                          max_num_episodes=10000, min_num_episodes=10, max_abserr=0.001, gamma=1.0,
                          iteration_prints=0,
                          max_episode_steps=1000,
                          epsilon=0.1, const_epsilon=True, epsilon_half_life=200,
                          alpha=0.25, const_alpha=True, alpha_half_life=200,
                          N_episodes_wo_decay=0)

print('_'*55)
score = gridworld.get_policy_score( policy, start_state_hash=None, step_limit=1000)
print('Policy Score =', score, ' = (r_sum, n_steps, msg)')


print( gridworld.get_info() )

episode = make_episode( gridworld.start_state_hash, policy, gridworld, gridworld.terminal_set )

epi_summ_print(episode, policy, gridworld, show_rewards=False,
               show_env_states=True, none_str='*')

