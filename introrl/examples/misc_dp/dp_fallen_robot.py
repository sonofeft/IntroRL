
from introrl.dp_funcs.dp_value_iter import dp_value_iteration
from introrl.dp_funcs.dp_policy_iter import dp_policy_iteration
from introrl.policy import Policy
from introrl.state_values import StateValues

from introrl.mdp_data.fallen_3state_robot import get_robot

robot = get_robot()

do_VI = 0
if do_VI:
    print('_____________ Value Iteration ________________')
else:
    print('_____________ Policy Iteration ________________')


for gamma in (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999):
    
    if do_VI:
        policy, sv = dp_value_iteration( robot, do_summ_print=False, fmt_V='%.1f',
                                         max_iter=1000, err_delta=0.001, 
                                         gamma=gamma)
    else:

        policy = Policy( environment=robot )
        policy.set_policy_from_piD( robot.get_default_policy_desc_dict() )

        sv = StateValues( robot )
        sv.init_Vs_to_zero()
        
        dp_policy_iteration(policy, sv,  do_summ_print=False,
                            max_iter=1000, err_delta=0.001, gamma=gamma)
        
    print('gamma=%5g'%gamma, 
          '  Fallen=', policy.get_single_action('Fallen'),
          '  Moving=', policy.get_single_action('Moving'),
          '  Standing=', policy.get_single_action('Standing'),
          '  Fallen=', '%g'%sv.VsD['Fallen'],
          '  Moving=', '%g'%sv.VsD['Moving'],
          '  Standing=', '%g'%sv.VsD['Standing'])
                          
print( robot.get_info() )
