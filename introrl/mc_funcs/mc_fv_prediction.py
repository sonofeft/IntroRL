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
from introrl.agent_supt.episode import Episode
from introrl.agent_supt.episode_maker import make_episode

def mc_first_visit_prediction( policy, state_value_ave, first_visit=True, 
                               do_summ_print=True, showRunningAve=False,
                               max_episode_steps=10000,
                               max_num_episodes=1000, min_num_episodes=10, 
                               max_abserr=0.001, gamma=0.9):
    """
    ... GIVEN A POLICY TO EVALUATE  apply Monte Carlo First Visit Prediction
    
    Use Episode Discounted Returns to find V(s), State-Value Function
    
    Terminates when abserr < max_abserr
    
    Assume that V(s), state_value_ave, has been initialized prior to call.
    (Note tht the StateValues object has a reference to the Environment object)
    
    Assume environment attached to policy will have method "get_any_action_state_hash"
    in order to begin at any start state.
    
    state_value_ave WILL BE CHANGED... policy WILL NOT.
    """
    
    # ==> Note: the reference to Environment object as "state_value_ave.environment"
    Env = state_value_ave.environment
    episode = Episode( Env.name + ' Episode' )
    
    if do_summ_print:
        print('=============== EVALUATING THE FOLLOWING POLICY ====================')
        policy.summ_print( verbosity=0, environment=Env, 
                   show_env_states=False, none_str='*')
                   
    s = 'Starting a Maximum of %i Monte Carlo All-Start-State Iterations\nGamma = %g'%(max_num_episodes, gamma)
    banner(s, banner_char='', leftMargin=0, just='center')
    
    keep_looping = True
       
    # value-iteration stopping criteria
    
    progress_str = ''
    num_episodes = 0
    
    while (num_episodes<=max_num_episodes-1) and keep_looping:
        
        keep_looping = False
        abserr = 0.0 # calculated below as part of termination criteria
        
        # policy evaluation 
        for start_hash in Env.iter_all_action_states( randomize=True ):
            
            # break from inner loop if max_num_episodes is hit.
            if num_episodes >= max_num_episodes:
                break
        
            make_episode(start_hash, policy, Env, Env.terminal_set, episode=episode,
                         max_steps=max_episode_steps, eps_greedy=None)
            
            num_episodes += 1
            
            for dr in episode.get_rev_discounted_returns( gamma=gamma, 
                                                          first_visit=first_visit, 
                                                          visit_type='S'):
                (s_hash, a_desc, reward, sn_hash, G) = dr
                state_value_ave.add_val( s_hash, G)
        
        abserr = state_value_ave.get_biggest_action_state_err()
        if abserr > max_abserr:
            keep_looping = True
            
        if num_episodes < min_num_episodes:
            keep_looping = True # must loop for min_num_episodes at least
                    
        pc_done = 100.0 * float(num_episodes) / float(max_num_episodes)
        out_str = '%i%%'%( 5*(int(pc_done/5.0) ) )
        if out_str != progress_str:
            print(out_str, end=' ')
            progress_str = out_str
            
    if do_summ_print:
        s = ''
        if num_episodes >= max_num_episodes:
            s = '   (NOTE: STOPPED ON MAX-ITERATIONS)'

        print( 'Exited MC First-Visit Policy Evaluation', s )
        print( '   num episodes   =', num_episodes, ' (min limit=%i)'%min_num_episodes, ' (max limit=%i)'%max_num_episodes )
        print( '   gamma          =', gamma )
        print( '   estimated err  =', abserr )
        print( '   Error limit    =', max_abserr )
    
        state_value_ave.summ_print( showRunningAve=showRunningAve, show_states=True)

    return abserr

if __name__ == "__main__": # pragma: no cover
    
    from introrl.policy import Policy
    from introrl.mdp_data.simple_grid_world import get_gridworld    
    from introrl.agent_supt.state_value_run_ave_coll import StateValueRunAveColl
    
    gridworld = get_gridworld()
    pi = Policy(  environment=gridworld  )
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    
    sv = StateValueRunAveColl( gridworld )
    #sv.init_Vs_to_zero() # done when StateValues is created.
    
    mc_first_visit_prediction( pi, sv, max_num_episodes=1000, max_abserr=0.001, gamma=0.9)
    
    
