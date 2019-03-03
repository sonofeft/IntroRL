from introrl.dp_funcs.dp_policy_eval import dp_policy_evaluation
from introrl.black_box_sims.random_walk_1000 import RandomWalk_1000Simulation
from introrl.policy import Policy
from introrl.state_values import StateValues
from introrl.agent_supt.model import Model
from introrl.environments.env_baseline import EnvBaseline
from introrl.utils import pickle_esp



RW = RandomWalk_1000Simulation()

    
model = Model( RW, build_initial_model=True )
model.collect_transition_data( num_det_calls=100, num_stoic_calls=10000 )
print('Model Built')
# build an EnvBaseline from the Simulation
env = EnvBaseline( s_hash_rowL=RW.s_hash_rowL, 
                   x_axis_label=RW.x_axis_label, 
                   y_axis_label=RW.y_axis_label )
model.add_all_data_to_an_environment( env )

policy = Policy( environment=env )
policy.intialize_policy_to_equiprobable( env=env )

state_value = StateValues( env )
state_value.init_Vs_to_zero()

dp_policy_evaluation(policy, state_value, 
                    do_summ_print=True, 
                    max_iter=1000, err_delta=0.0001, gamma=1.0)

pickle_esp.save_to_pickle_file( fname='random_walk_1000_PI_eval', env=env, 
                             state_values=state_value, policy=policy)
