from introrl.dp_funcs.dp_policy_eval import dp_policy_evaluation
from introrl.policy import Policy
from introrl.state_values import StateValues
from introrl.mdp_data.sutton_ex4_1_grid import get_gridworld

gridworld = get_gridworld()

pi = Policy( environment=gridworld )
pi.intialize_policy_to_equiprobable( env=gridworld )

sv = StateValues( gridworld )
sv.init_Vs_to_zero()

dp_policy_evaluation( pi, sv, max_iter=1000, err_delta=0.001, gamma=1., fmt_V='%.1f')

#sv.summ_print( fmt_V='%.3f', show_states=False )
pi.summ_print(  environment=gridworld, verbosity=0, show_env_states=False  )

#print( gridworld.get_info() )