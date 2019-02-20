import time
from introrl.dp_funcs.dp_value_iter import dp_value_iteration
from introrl.environments.env_baseline import EnvBaseline
from introrl.agent_supt.model import Model
from introrl.utils import pickle_esp
from introrl.black_box_sims.blackjack_sim import BlackJackSimulation

start_time = time.time()

BJ = BlackJackSimulation()
get_sim = Model( BJ, build_initial_model=True )

get_sim.collect_transition_data( num_det_calls=50, num_stoic_calls=100000 )

BJ.layout.s_hash_print()

get_sim.num_calls_layout_print()
get_sim.min_num_calls_layout_print()

print('got sim data')
print('_'*55)

env = EnvBaseline( s_hash_rowL=BJ.s_hash_rowL, 
                   x_axis_label=BJ.x_axis_label, 
                   y_axis_label=BJ.y_axis_label )
                   
get_sim.add_all_data_to_an_environment( env )


print('built environment')
print('_'*55)

#env.summ_print()
policy, state_value = dp_value_iteration( env, do_summ_print=True, fmt_V='%.1f', fmt_R='%.1f',
                                          max_iter=1000, err_delta=0.0001, 
                                          gamma=0.9, iteration_prints=10)
                              
policy.save_diagram( BJ, inp_colorD=None, save_name='dp_blackjack_policy',
                     show_arrows=False, scale=0.5, h_over_w=0.8,
                     show_terminal_labels=False)

print( 'Total Time =',time.time() - start_time )

pickle_esp.save_to_pickle_file( fname='dp_soln_to_blackjack', 
                                env=env, state_values=state_value, policy=policy)

