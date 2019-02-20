#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from introrl.utils.banner import banner
from introrl.agent_supt.alpha_calc import Alpha

def td0_prediction( policy, state_value_coll,  all_start_states=False,
                               do_summ_print=True, show_last_change=True,
                               alpha=0.1, const_alpha=True, alpha_half_life=200,
                               max_num_episodes=1000, min_num_episodes=10, max_abserr=0.001, gamma=1.0,
                               result_list='abserr', true_valueD=None,
                               value_snapshot_loopL=None): # if input, save V(s) snapshot at iteration steps indicated
    """
    ... GIVEN A POLICY TO EVALUATE  apply TD(0), Temperal Difference(0) Prediction
    
    Terminates when abserr < max_abserr
    
    Assume that Q(s,a), action_value_coll, has been initialized prior to call.
    (Note tht the StateValues object has a reference to the Environment object)
    
    Assume environment attached to policy will have method "get_any_action_state_hash"
    in order to begin at any start state.
    
    action_value_coll WILL BE CHANGED... policy WILL NOT.
    """
    
    resultL = [] # based on result_list, can be "rms" or "abserr"
    value_snapD = {} # index=loop counter, value=dict of {s_hash:Vs, ...}
    
    # ==> Note: the reference to Environment object as "state_value_coll.environment"
    Env = state_value_coll.environment
    
    alpha_obj = Alpha( alpha=alpha, const_alpha=const_alpha, half_life=alpha_half_life )
    
    if do_summ_print:
        print('=============== TD(0) EVALUATING THE FOLLOWING POLICY ====================')
        policy.summ_print( verbosity=0, environment=Env, 
                   show_env_states=False, none_str='*')
    
    if  all_start_states:
        s = 'Starting a Maximum of %i TD(0) All-Start-State Episodes\nGamma = %g'%\
            (max_num_episodes, gamma)
        start_stateL = [s_hash for s_hash in Env.iter_all_action_states()]
    else:
        s = 'Starting a Maximum of %i TD(0) Episodes from state "%s"\nGamma = %g'%\
            (max_num_episodes, str(Env.start_state_hash), gamma)
        start_stateL = [ Env.start_state_hash ]

    banner(s, banner_char='', leftMargin=0, just='center')
    
    num_episodes = 0
    keep_looping = True
       
    # value-iteration stopping criteria
    
    progress_str = ''
    while (num_episodes<=max_num_episodes-1) and keep_looping:
        
        keep_looping = False
        abserr = 0.0 # calculated below as part of termination criteria
        
        # policy evaluation 
        for start_hash in start_stateL:
            num_episodes += 1
            if num_episodes > max_num_episodes:
                break
            
            s_hash = start_hash
            a_desc = policy.get_single_action( s_hash )
            for _ in range( max_num_episodes ):
        
                sn_hash, reward = Env.get_action_snext_reward( s_hash, a_desc ) # prob-weighted choice
                
                state_value_coll.td0_update( s_hash=s_hash, alpha=alpha_obj(), 
                                                         gamma=gamma, sn_hash=sn_hash, 
                                                         reward=reward)
                
                if (sn_hash in Env.terminal_set) or (sn_hash is None):
                    break
                
                # get ready for next step
                s_hash = sn_hash
                
                a_desc = policy.get_single_action( s_hash )
                if a_desc is None:
                    print('a_desc is None for policy.get_single_action( "%s" ) ='%\
                          str(s_hash), a_desc)
                
        abserr = state_value_coll.get_biggest_action_state_err()
        if abserr > max_abserr:
            keep_looping = True
            
        if num_episodes < min_num_episodes:
            keep_looping = True # must loop for min_num_episodes at least
            
        pc_done = 100.0 * float(num_episodes) / float(max_num_episodes)
        out_str = '%i%%'%( 5*(int(pc_done/5.0) ) )
        if out_str != progress_str:
            print(out_str, end=' ')
            progress_str = out_str
            
        if result_list=='rms':
            resultL.append( state_value_coll.calc_rms_error(true_valueD) )
        if result_list=='abserr':
            resultL.append( abserr )
        else:
            pass # don't save anything to resultL
            
    if value_snapshot_loopL is not None and num_episodes in value_snapshot_loopL:
        value_snapD[num_episodes] = state_value_coll.get_snapshot()
            
    if do_summ_print:
        s = ''
        if num_episodes >= max_num_episodes:
            s = '   (NOTE: STOPPED ON MAX-ITERATIONS)'

        print( 'Exited MC Every-Visit Policy Evaluation', s )
        print( '   num_episodes   =', num_episodes, ' (min limit=%i)'%min_num_episodes, ' (max limit=%i)'%max_num_episodes )
        print( '   gamma          =', gamma )
        print( '   estimated err  =', abserr )
        print( '   Error limit    =', max_abserr )
    
        state_value_coll.summ_print( show_last_change=show_last_change, show_states=True)

    return resultL, value_snapD

if __name__ == "__main__": # pragma: no cover
    
    from introrl.policy import Policy
    from introrl.agent_supt.state_value_coll import StateValueColl
    
    from introrl.mdp_data.simple_grid_world import get_gridworld    
    gridworld = get_gridworld()
    pi = Policy(  environment=gridworld  )
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    
    sv = StateValueColl( gridworld )
    #sv.init_Vs_to_zero() # done when StateValues is created.
    
    td0_prediction( pi, sv, max_num_episodes=1000, max_abserr=0.001, gamma=1.0)
    
    
