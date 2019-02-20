#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object


from introrl.utils.functions import argmax_vmax_dict
    
def dp_policy_improvement( policy, state_value, gamma=0.9, 
                      do_summ_print=True, max_iter=1000):
    """
    ... GIVEN STATE-VALUES ...  apply State-Value Policy Improvement
    
    Use Policy-Improvement to find best policy for current V(s) values
    
    Terminates when policy is stable.
    
    Assume that V(s), state_value, has been initialized prior to call.
    (Note tht the StateValues object has a reference to the Environment object)
    
    policy WILL BE CHANGED... state_value WILL NOT.
    """
    
    loop_counter = 0
    is_stable = False
    made_changes = False
           
    # Note: the reference to Environment object as "state_value.environment"
    Env = state_value.environment
    
    while (loop_counter<max_iter) and (not is_stable):
        loop_counter += 1
        is_stable = True
        
        # policy improvement
        for s_hash in policy.iter_all_policy_states():
            old_action = policy.get_single_action( s_hash )
            
            VsD = {} # will hold: index=a_desc, value=V(s) for all transitions of a_desc from s_hash
            for a_desc, a_prob in policy.iter_policy_ap_for_state( s_hash, incl_zero_prob=True):
                VsD[a_desc] = 0.0
                for sn_hash, t_prob, reward in \
                    Env.iter_next_state_prob_reward(s_hash, a_desc, incl_zero_prob=False):
                    
                    # need to assume that a_prob==1.0
                    #VsD[a_desc] += t_prob * a_prob * ( reward + gamma * state_value(sn_hash) )
                    VsD[a_desc] += t_prob * ( reward + gamma * state_value(sn_hash) )

            # use pick_random_best=False to avoid subtle non-termination bug (see page 82)
            best_a_desc, best_a_val = argmax_vmax_dict( VsD, pick_random_best=False )
            
            if best_a_desc != old_action:
                is_stable = False
                made_changes = True # returned to caller
            
            policy.set_sole_action( s_hash, best_a_desc)
    
    if do_summ_print:
        s = ''
        if loop_counter >= max_iter:
            s = '   (NOTE: STOPPED ON MAX-ITERATIONS)'
        print( '=========================' + '='*len(s) )
        print( 'Exited Policy Improvement', s )
        print( '   iterations    =', loop_counter, ' (limit=%i)'%max_iter )
        print( '   gamma         =', gamma )
        print( '=========================' + '='*len(s) )
    
        state_value.summ_print()
        
    return made_changes

if __name__ == "__main__": # pragma: no cover
    
    from introrl.policy import Policy
    from introrl.state_values import StateValues
    from introrl.dp_funcs.dp_policy_eval import dp_policy_evaluation
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    pi = Policy(  environment=gridworld  )
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    
    print('-'*55)
    sv = StateValues( gridworld )
    sv.init_Vs_to_zero()
    
    dp_policy_evaluation( pi, sv, max_iter=1000, err_delta=0.001, gamma=0.9, do_summ_print=False)

    print('-'*55)
    pi_2 = Policy(  environment=gridworld  )
    pi_2.intialize_policy_to_random( env=gridworld )
    print('-------- Random Policy Prior to Improvement ----------')
    pi_2.summ_print( verbosity=0 )
    
    # sv should be optimum for the "pi" policy... see if "pi_2" is changed to "pi"
    dp_policy_improvement( pi_2, sv, gamma=0.9, do_summ_print=True, max_iter=1000)
    
    print('-------- Random Policy AFTER Improvement ----------')
    pi_2.summ_print(  environment=gridworld, verbosity=0, show_env_states=False  )
    print('-------- Default gridworld policy ----------')
    pi.summ_print(  environment=gridworld, verbosity=0, show_env_states=False  )
    
