from introrl.dp_funcs.dp_value_iter import dp_value_iteration
from introrl.agent_supt.episode_maker import make_episode
from introrl.agent_supt.episode_summ_print import epi_summ_print

from introrl.mdp_data.windy_gridworld import get_gridworld

gridworld = get_gridworld( step_reward=-1 )

policy, state_value = dp_value_iteration( gridworld, do_summ_print=True,fmt_V='%.1f',
                                          max_iter=1000, err_delta=0.001, 
                                          gamma=1.0)

print('_'*55)
score = gridworld.get_policy_score( policy, start_state_hash=None, step_limit=1000)
print('Policy Score =', score, ' = (r_sum, n_steps, msg)')


print( gridworld.get_info() )

episode = make_episode( gridworld.start_state_hash, policy, gridworld, gridworld.terminal_set )

epi_summ_print(episode, policy, gridworld, show_rewards=False,
               show_env_states=True, none_str='*')

