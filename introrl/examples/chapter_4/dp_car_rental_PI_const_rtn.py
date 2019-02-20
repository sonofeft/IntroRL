
from introrl.dp_funcs.dp_policy_iter import dp_policy_iteration
from introrl.policy import Policy
from introrl.state_values import StateValues
from introrl.mdp_data.car_rental_const_rtn import get_env
from introrl.utils import pickle_esp

env = get_env()

policy = Policy( environment=env )
policy.intialize_policy_to_random( env=env )

state_value = StateValues( env )
state_value.init_Vs_to_zero()


dp_policy_iteration(policy, state_value, 
                    do_summ_print=True, show_start_policy=True,
                    max_iter=1000, err_delta=0.0001, gamma=0.9)

pickle_esp.save_to_pickle_file( fname='dp_car_rental_PI_const_rtn', env=env, 
                             state_values=state_value, policy=policy)
                             
state_value.summ_print( fmt_V='%.1f')


policy.save_diagram( env, inp_colorD={'5':'r', '4':'g', '3':'b', '2':'c', '1':'y', '0':'w', '-5':'r', '-4':'g', '-3':'b', '-2':'c', '-1':'y'}, 
                 save_name='car_rental_const_rtn', show_arrows=False)
