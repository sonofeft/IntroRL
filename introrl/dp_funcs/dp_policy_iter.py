#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object


from introrl.dp_funcs.dp_policy_improve import dp_policy_improvement
from introrl.dp_funcs.dp_policy_eval import dp_policy_evaluation

def dp_policy_iteration(policy, state_value, do_summ_print=True, 
                        show_start_policy=True,
                        show_each_policy_change=False,
                        max_iter=1000, err_delta=0.001, gamma=0.9):
    """
    ... GIVEN A POLICY: EVALUATE IT ONCE, THEN IMPROVE THE POLICY ONCE ...
    First do State-Value Policy Evaluation:
        Use Policy-Evaluation to find V(s), State-Value Function
        Terminates when delta < err_delta * VI_STOP_CRITERIA
    
    ....... THEN ........    
    Do State-Value Policy Improvement
        Use Policy-Improvement to find best policy for current V(s) values
        Terminates when policy is stable.
    
    Assume that V(s), state_value, has been initialized prior to call.
    (Note tht the StateValues object has a reference to the Environment object)
    
    BOTH policy AND state_value WILL BE CHANGED
    """

    if show_start_policy:
        print( 'Starting Policy-Iteration'.center(60, '#' ) )
        print('  --> Initial Policy BEFORE POLICY ITERATION <--')
        policy.summ_print(  environment=state_value.environment, verbosity=0, show_env_states=False  )
    
    made_changes = True
    max_delta = 1.0E6
    counter = 0
    while made_changes or (max_delta>err_delta):
        counter += 1
        
        max_delta = dp_policy_evaluation( policy, state_value, do_summ_print=False,
                                          max_iter=max_iter, err_delta=err_delta, gamma=gamma)
        
        print('#%i) policy iteration: max_delta ='%counter,max_delta)
            
        
        made_changes = dp_policy_improvement( policy, state_value, gamma=gamma, 
                                             do_summ_print=False, max_iter=max_iter)
                                             
        if made_changes and show_each_policy_change:
            policy.summ_print(  environment=state_value.environment, verbosity=0, show_env_states=False  )

    if do_summ_print:
        state_value.summ_print( fmt_V='%g', none_str='*', show_states=True)
        
        print('  --> Final Policy AFTER POLICY ITERATION <--')
        policy.summ_print(  environment=state_value.environment, verbosity=0, show_env_states=False  )

if __name__ == "__main__": # pragma: no cover
    import sys
    from introrl.policy import Policy
    from introrl.state_values import StateValues
    from introrl.dp_funcs.dp_policy_eval import dp_policy_evaluation
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    pi = Policy( environment=gridworld )
    
    #pi.intialize_policy_to_equiprobable(env=gridworld)
    pi.intialize_policy_to_random(env=gridworld)
    #pi.learn_all_states_and_actions_from_env( gridworld )
    
    #pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    
    # change one action from gridworld default
    pi.set_sole_action( (1,0), 'D') # is 'U' in default
        
    sv = StateValues( gridworld )
    sv.init_Vs_to_zero()
    
    dp_policy_iteration(pi, sv, do_summ_print=True, show_each_policy_change=True,
                        max_iter=1000, err_delta=0.001, gamma=0.9)
