#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from introrl.policy import Policy
from introrl.state_values import StateValues
from introrl.utils.functions import argmax_vmax_dict, multi_argmax_vmax_dict
    
def dp_value_iteration( environment, allow_multi_actions=False,
                        do_summ_print=True, fmt_V='%g', fmt_R='%g',
                        max_iter=1000, err_delta=0.001, gamma=0.9,
                        iteration_prints=0):
    """
    ... GIVEN AN ENVIRONMENT ... 
    apply Value Iteration to find the OPTIMAL POLICY
    
    Returns: policy and state_value objects
    
    Terminates when delta < err_delta * VI_STOP_CRITERIA
    
    CREATES BOTH policy AND state_value OBJECTS.
    
    If allow_multi_actions is True, policy will include all actions 
    within err_delta of best action.
    """
    
    # create Policy and StateValues objects
    policy = Policy( environment=environment )
    policy.intialize_policy_to_random(env=environment)
        
    state_value = StateValues( environment )
    state_value.init_Vs_to_zero() # Terminal states need to be 0.0
    #state_value.summ_print()
    
    # set counter and flag
    loop_counter = 0
    all_done = False
       
    # value-iteration stopping criteria
    # if gamme==1.0 value iteration will never stop SO limit to gamma==0.999 stop criteria
    #  (VI terminates if delta < err_delta * VI_STOP_CRITERIA)
    #  (typically err_delta = 0.001)
    
    VI_STOP_CRITERIA = max((1.0-gamma) / gamma, (1.0-0.999)/0.999) 
    error_limit = err_delta * VI_STOP_CRITERIA
        
    while (loop_counter<max_iter) and (not all_done):
        loop_counter += 1
        all_done = True
        delta = 0.0 # used to calc largest change in state_value
        
        for s_hash in policy.iter_all_policy_states():
            VsD = {} # will hold: index=a_desc, value=V(s) for all transitions of a_desc from s_hash
            
            # MUST include currently zero prob actions
            for a_desc, a_prob in policy.iter_policy_ap_for_state( s_hash, incl_zero_prob=True):
                calcd_v = 0.0
                
                for sn_hash, t_prob, reward in \
                    environment.iter_next_state_prob_reward(s_hash, a_desc, incl_zero_prob=False):
                    
                    calcd_v += t_prob * ( reward + gamma * state_value(sn_hash) )
            
                VsD[a_desc] = calcd_v
            
            best_a_desc, best_a_val = argmax_vmax_dict( VsD )
            delta = max( delta, abs(best_a_val - state_value(s_hash)) )
            state_value[s_hash] = best_a_val
            
        if delta > error_limit:
            all_done = False
        
        if iteration_prints and (loop_counter % iteration_prints == 0):
            print('Loop:%6i'%loop_counter,'  delta=%g'%delta)
        
    # Now that State-Values have been determined, set policy
    for s_hash in policy.iter_all_policy_states():
        VsD = {} # will hold: index=a_desc, value=V(s) for all transitions of a_desc from s_hash
        
        # MUST include zero prob actions
        for a_desc, a_prob in policy.iter_policy_ap_for_state( s_hash, incl_zero_prob=True):
            calcd_v = 0.0
            
            for sn_hash, t_prob, reward in \
                environment.iter_next_state_prob_reward(s_hash, a_desc, incl_zero_prob=False):
                
                calcd_v += t_prob * ( reward + gamma * state_value(sn_hash) )
        
            VsD[a_desc] = calcd_v
        
        if allow_multi_actions:
            best_a_list, best_a_val = multi_argmax_vmax_dict( VsD, err_delta=err_delta )
            
            policy.set_sole_action( s_hash, best_a_list[0]) # zero all other actions
            prob = 1.0 / len(best_a_list)
            for a_desc in best_a_list:
                policy.set_action_prob( s_hash, a_desc, prob=prob)
        else:
            best_a_desc, best_a_val = argmax_vmax_dict( VsD )
            policy.set_sole_action( s_hash, best_a_desc)
    
    
    if do_summ_print:
        s = ''
        if loop_counter >= max_iter:
            s = '   (NOTE: STOPPED ON MAX-ITERATIONS)'

        print( 'Exited Value Iteration', s )
        print( '   iterations     =', loop_counter, ' (limit=%i)'%max_iter )
        print( '   measured delta =', delta )
        print( '   gamma          =', gamma )
        print( '   err_delta      =', err_delta )
        print( '   error limit    =',error_limit )
        print( '   STOP CRITERIA  =',VI_STOP_CRITERIA)
    
        state_value.summ_print( fmt_V=fmt_V )
        policy.summ_print(  environment=environment, verbosity=0, show_env_states=False  )
        
        environment.layout_print( vname='reward', fmt=fmt_R, show_env_states=False, none_str='*')
        
    return policy, state_value

if __name__ == "__main__": # pragma: no cover
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    
    policy, state_value = dp_value_iteration( gridworld, do_summ_print=True,
                                              max_iter=1000, err_delta=0.001, 
                                              gamma=0.9)
