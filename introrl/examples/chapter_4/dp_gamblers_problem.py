import matplotlib
import matplotlib.pyplot as plt

from introrl.dp_funcs.dp_value_iter import dp_value_iteration
from introrl.mdp_data.gamblers_problem import get_gambler

gambler = get_gambler(prob_heads=0.4)

policy, state_value = dp_value_iteration( gambler, allow_multi_actions=True,
                                          do_summ_print=True,fmt_V='%.4f',
                                          max_iter=1000, err_delta=0.00001, 
                                          gamma=1.0)
print( gambler.get_info() )

# --------------- plot logic -------------------
min_state_list = []
min_action_list = []

state_list = []
action_list = []
for i_state in range(1,100):
    aL = policy.get_list_of_all_action_desc_prob( i_state, incl_zero_prob=False)
    min_state_list.append( i_state - 0.5 )
    min_action_list.append( min([a for a,p in aL]) )

    min_state_list.append( i_state + 0.5 )
    min_action_list.append( min([a for a,p in aL]) )

    for a,p in aL:
        action_list.append( a )
        state_list.append( i_state )
    
fig, ax = plt.subplots()
plt.plot( min_state_list, min_action_list, '-' )
plt.title( 'Figure 4.3, Gamblers Optimum Policy' )
plt.xlabel('Capital')
plt.ylabel('Final Policy (stake)')

fig.savefig( 'figure_4_3_gamblers_policy.png' )

# --------------------------------
    
fig, ax = plt.subplots()
plt.plot( state_list, action_list, 's' )
plt.title( 'Figure 4.3, Gamblers Full Optimum Policy' )
plt.xlabel('Capital')
plt.ylabel('Final Policy (stake)')

fig.savefig( 'figure_4_3_gamblers_full_policy.png' )


plt.show()
