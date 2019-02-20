
from introrl.dp_funcs.dp_policy_iter import dp_policy_iteration
from introrl.policy import Policy
from introrl.state_values import StateValues
from introrl.utils import pickle_esp

env, state_value, policy = pickle_esp.read_pickle_file( fname='dp_car_rental_PI' )

diag_colorD = {'5':'r', '4':'g', '3':'b', '2':'c', '1':'y', '0':'w', 
               '-5':'r', '-4':'g', '-3':'b', '-2':'c', '-1':'y'}
                   
policy.save_diagram( env, inp_colorD=diag_colorD, save_name='car_rental_var_rtn_v3',
                     show_arrows=False, scale=0.25, h_over_w=0.8, do_show=True)

state_value.summ_print( fmt_V='%.1f')
