from introrl.dp_funcs.dp_value_iter import dp_value_iteration
from introrl.mdp_data.simple_grid_world import get_gridworld

gridworld = get_gridworld()

policy, state_value = dp_value_iteration( gridworld, do_summ_print=True,
                                          max_iter=1000, err_delta=0.001, 
                                          gamma=0.9)
print( gridworld.get_info() )
