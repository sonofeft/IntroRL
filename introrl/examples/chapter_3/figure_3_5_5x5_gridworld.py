
from introrl.dp_funcs.dp_value_iter import dp_value_iteration
from introrl.environments.env_baseline import EnvBaseline

from introrl.mdp_data.sutton_5x5_gridworld import get_gridworld
gridworld = get_gridworld()
gridworld.name = 'Figure 3.5, 5x5 Grid Value Iteration'

policy, state_value = dp_value_iteration( gridworld, do_summ_print=True,fmt_V='%.1f',
                                          max_iter=1000, err_delta=0.001, 
                                          gamma=0.9, allow_multi_actions=True)

policy.save_diagram( gridworld, inp_colorD=None, save_name='figure_3_5_policy',
                     show_arrows=True, scale=0.8, h_over_w=0.8, do_show=True)
