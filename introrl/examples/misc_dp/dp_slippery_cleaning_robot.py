
from introrl.dp_funcs.dp_value_iter import dp_value_iteration
from introrl.dp_funcs.dp_policy_eval import dp_policy_evaluation
from introrl.policy import Policy
from introrl.state_values import StateValues
    
from introrl.mdp_data.slippery_cleaning_robot import get_robot

gridworld = get_robot()

if 1:
    policy, state_value = dp_value_iteration( gridworld, do_summ_print=True,fmt_V='%.3f',
                                              max_iter=1000, err_delta=0.001, 
                                              gamma=1.0)
    
    print('_'*55)
    score = gridworld.get_policy_score( policy, start_state_hash=None, step_limit=1000)
    print('Policy Score =', score, ' = (r_sum, n_steps, msg)')

else:

    pi = Policy( environment=gridworld )
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )

    sv = StateValues( gridworld )
    sv.init_Vs_to_zero()

    dp_policy_evaluation( pi, sv, max_iter=1000, err_delta=0.001, gamma=.985)

    #sv.summ_print( fmt_V='%.3f', show_states=False )
    pi.summ_print(  environment=gridworld, verbosity=0, show_env_states=False  )



print( gridworld.get_info() )


print("""
NOTE: the web site answer is WRONG... the optimum policy is actually
     ___ Simple Grid World Policy Summary ___
                   R   R   R   * 
                   U   *   U   * 
                   U   L   L   D 
     _______________ Actions ________________

...........................NOT..........................

     ___     The published answer of:    ___
                   R   R   R   * 
                   U   *   U   * 
                   U   L   L   L
     _______________ Actions ________________

The Lower Right Move Avoids the -1 penalty by going D until the 10%
chance of actually going L
""")
