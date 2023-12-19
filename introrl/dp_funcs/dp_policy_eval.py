#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

    
def dp_policy_evaluation( policy, state_value, do_summ_print=True, fmt_V='%g',
                          max_iter=1000, err_delta=0.001, gamma=0.9):
    """
    ... GIVEN A POLICY TO EVALUATE  apply State-Value Policy Evaluation 
    
    Use Policy-Evaluation to find V(s), State-Value Function
    
    Terminates when delta < err_delta * VI_STOP_CRITERIA
    
    Assume that V(s), state_value, has been initialized prior to call.
    (Note tht the StateValues object has a reference to the Environment object)
    
    state_value WILL BE CHANGED... policy WILL NOT.
    
    This code takes the state_values all the way to their final values for this policy.
    More general policy evaluation only goes part-way before improving the policy.
    """
    
    loop_counter = 0
    all_done = False
       
    # value-iteration stopping criteria
    # if gamme==1.0 value iteration will never stop SO limit to gamma==0.999 stop criteria
    #  (VI terminates if delta < err_delta * VI_STOP_CRITERIA)
    #  (typically err_delta = 0.001)
    VI_STOP_CRITERIA = max((1.0-gamma) / gamma, (1.0-0.999)/0.999) 
    error_limit = err_delta * VI_STOP_CRITERIA
    
    # ==> Note: the reference to Environment object as "state_value.environment"
    Env = state_value.environment
    max_delta = 0.0
    
    while (loop_counter<max_iter) and (not all_done):
        loop_counter += 1
        all_done = True
        delta = 0.0 # used to calc largest change in state_value
        
        # policy evaluation 
        for s_hash in policy.iter_all_policy_states():
            
            calcd_v = 0.0
            for a_desc, a_prob in policy.iter_policy_ap_for_state( s_hash, incl_zero_prob=False):
                
                for sn_hash, t_prob, reward in \
                    Env.iter_next_state_prob_reward(s_hash, a_desc, incl_zero_prob=False):
                    
                    calcd_v += t_prob * a_prob * ( reward + gamma * state_value(sn_hash) )
            
            delta = max( delta, abs(calcd_v - state_value(s_hash)) )
            if delta > error_limit:
                all_done = False
                max_delta = max(max_delta, delta) # returned to caller
            
            state_value[s_hash] = calcd_v
    
    if do_summ_print:
        s = ''
        if loop_counter >= max_iter:
            s = '   (NOTE: STOPPED ON MAX-ITERATIONS)'

        print( 'Exited Policy Evaluation', s )
        print( '   iterations     =', loop_counter, ' (limit=%i)'%max_iter )
        print( '   measured delta =', delta )
        print( '   gamma          =', gamma )
        print( '   err_delta      =', err_delta )
        print( '   error limit    =',error_limit )
        print( '   STOP CRITERIA  =',VI_STOP_CRITERIA)
    
        state_value.summ_print( fmt_V=fmt_V )

    return max_delta

if __name__ == "__main__": # pragma: no cover
    
    from introrl.policy import Policy
    from introrl.mdp_data.simple_grid_world import get_gridworld
    from introrl.state_values import StateValues
    
    gridworld = get_gridworld()
    pi = Policy(  environment=gridworld  )
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    
    sv = StateValues( gridworld )
    #sv.init_Vs_to_zero() # done when StateValues is created.
    
    dp_policy_evaluation( pi, sv, max_iter=1000, err_delta=0.001, gamma=0.9)
    
    print()
    print( pi.make_dict_of_policy() )
    pi.save_diagram( gridworld, inp_colorD=None, pad=0.1, save_name='', 
                     show_arrows=True, do_show=True, scale=1.0, h_over_w=1.0,
                     show_terminal_labels=True)
    
