
from introrl.environments.env_baseline import EnvBaseline

from introrl.dp_funcs.dp_policy_eval import dp_policy_evaluation
from introrl.policy import Policy
from introrl.state_values import StateValues

from introrl.mdp_data.sutton_5x5_gridworld import get_gridworld
gridworld = get_gridworld()
gridworld.name = 'Figure 3.2, 5x5 Grid Policy Evaluation'

pi = Policy( environment=gridworld )
pi.intialize_policy_to_equiprobable( env=gridworld )

sv = StateValues( gridworld )
sv.init_Vs_to_zero()

dp_policy_evaluation( pi, sv, max_iter=1000, err_delta=0.0001, gamma=0.9, fmt_V='%.1f')

#sv.summ_print( fmt_V='%.3f', show_states=False )
#pi.summ_print(  environment=gridworld, verbosity=0, show_env_states=False  )
