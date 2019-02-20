#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
import random

from introrl.policy import Policy
from introrl.agent_supt.action_value_run_ave_coll import ActionValueRunAveColl

from introrl.utils.banner import banner
from introrl.agent_supt.episode import Episode
from introrl.agent_supt.episode_maker import make_episode
from introrl.agent_supt.epsilon_calc import EpsilonGreedy

def mc_epsilon_greedy( environment, initial_policy='default', # can be 'default', 'random', policy_dictionary
                       read_pickle_file='', 
                       save_pickle_file='',
                       use_list_of_start_states=True, # use list OR single start state of environment.
                       iter_all_start_actions=False, # pick random or iterate all starting actions
                       first_visit=True, 
                       do_summ_print=True, showRunningAve=False, fmt_Q='%g', fmt_R='%g',
                       show_initial_policy=True,
                       max_num_episodes=1000, min_num_episodes=10, max_abserr=0.001, gamma=0.9,
                       iteration_prints=0,
                       max_episode_steps=10000,
                       epsilon=0.1, const_epsilon=True, half_life=200,
                       N_episodes_wo_decay=0):
    """
    ... GIVEN AN ENVIRONMENT ... 
    apply Monte Carlo Exploring Starts to find the OPTIMAL POLICY
    
    Returns: Policy and ActionValueRunAveColl objects
    
    Use Episode Discounted Returns to find Q(s,a), Action-Value Function
    
    Terminates when abserr < max_abserr
    
    Assume that Q(s,a), action_value_ave, has been initialized prior to call.
    
    Assume environment attached to policy will have method "get_any_action_state_hash"
    in order to begin at any action state.
    
    CREATES BOTH policy AND action_value OBJECTS.
    """
    
    eps_greedy = EpsilonGreedy(epsilon=epsilon, const_epsilon=const_epsilon, half_life=half_life,
                               N_episodes_wo_decay=N_episodes_wo_decay)
    
    # create Policy and ActionValueRunAveColl objects
    policy = Policy( environment=environment )
    if initial_policy=='default':
        print('Initializing Policy to "default" in mc_epsilon_greedy')
        policy.learn_a_legal_action_from_env( env=environment )
        policy.set_policy_from_piD( environment.get_default_policy_desc_dict() )
    elif initial_policy=='random':
        print('Initializing Policy to "random" in mc_epsilon_greedy')
        policy.intialize_policy_to_random(env=environment)
    else:
        print('Initializing Policy to "custom policy" in mc_epsilon_greedy')
        policy.learn_a_legal_action_from_env( env=environment )
        policy.set_policy_from_piD( initial_policy )


    action_value_ave = ActionValueRunAveColl( environment )
    action_value_ave.init_Qsa_to_zero() # Terminal states w/o an action are NOT included
    #action_value_ave.summ_print()

    if read_pickle_file:
        policy.init_from_pickle_file( read_pickle_file )
        action_value_ave.init_from_pickle_file( read_pickle_file )
    
    if do_summ_print:
        if show_initial_policy:
            print('=============== STARTING WITH THE INITIAL POLICY ====================')
            policy.summ_print( verbosity=0, environment=environment, 
                       show_env_states=False, none_str='*')

        print('================== EPSILON GREEDY DEFINED AS ========================')
        eps_greedy.summ_print()
                   
    s = 'Starting a Maximum of %i Monte Carlo Epsilon Greedy\nfor "%s" with Gamma = %g'%\
        (max_num_episodes, environment.name, gamma)
    banner(s, banner_char='', leftMargin=0, just='center')
    
    
    # create an Episode object for getting returns
    episode = Episode( environment.name + ' Episode' )
    
    # set counter and flag
    num_episodes = 0
    keep_looping = True
    
    limited_start_stateL = environment.limited_start_state_list()
    
    progress_str = ''
    while (num_episodes<=max_num_episodes-1) and keep_looping :
        
        keep_looping = False
        abserr = 0.0 # calculated below as part of termination criteria
        Nterminal_episodes = set()
        
        # Iterate over a list of known possible start states
        if use_list_of_start_states:
            loop_stateL = limited_start_stateL
            random.shuffle( loop_stateL )
        else:
            #loop_stateL = [ random.choice( limited_start_stateL ) ]
            loop_stateL = [ environment.start_state_hash ]
        
        for start_hash in loop_stateL:
            
            if iter_all_start_actions:# Iterate over ALL ACTIONS of start_hash
                a_descL = environment.get_state_legal_action_list( start_hash )
            else:
                a_desc = policy.get_single_action( start_hash )
                # if not iterating all actions, make sure first action has eps-greedy applied
                a_desc = eps_greedy( a_desc, 
                                     environment.get_state_legal_action_list( start_hash ) )
                a_descL = [ a_desc ]
            # randomize action order
            random.shuffle( a_descL )
            
            for a_desc in a_descL:
                
                # break from inner loop if max_num_episodes is hit.
                if num_episodes >= max_num_episodes:
                    break
                
                make_episode(start_hash, policy, 
                             environment, environment.terminal_set, 
                             episode=episode, first_a_desc=a_desc,
                             max_steps=max_episode_steps, eps_greedy=eps_greedy)
                eps_greedy.inc_N_episodes()
                num_episodes += 1
                
                if episode.is_done():
                    Nterminal_episodes.add( start_hash )
            
                for dr in episode.get_rev_discounted_returns( gamma=gamma, 
                                                              first_visit=first_visit, 
                                                              visit_type='SA'):
                    # look at each step from episode and calc average Q(s,a)
                    (s, a, r, sn, G) = dr
                    action_value_ave.add_val( s, a, G)
                    
                    aL = environment.get_state_legal_action_list( s )
                    if aL:
                        best_a_desc, best_a_val = aL[0], float('-inf')
                        bestL = [best_a_desc]
                        for a in aL:
                            q = action_value_ave.get_ave( s, a )
                            if q > best_a_val:
                                best_a_desc, best_a_val = a, q
                                bestL = [ a ]
                            elif q == best_a_val:
                                bestL.append( a )
                        best_a_desc = random.choice( bestL )
                        policy.set_sole_action(s, best_a_desc)
                
        abserr = action_value_ave.get_biggest_action_state_err()
        if abserr > max_abserr:
            keep_looping = True
            
        if num_episodes < min_num_episodes:
            keep_looping = True # must loop for min_num_episodes at least
        
        pc_done = 100.0 * float(num_episodes) / float(max_num_episodes)
        out_str = '%3i%%'%( 5*(int(pc_done/5.0) ) )
        if out_str != progress_str:
            score = environment.get_policy_score( policy=policy, start_state_hash=None, step_limit=1000)
            print(out_str, ' score=%s'%str(score), ' = (r_sum, n_steps, msg)', end=' ')
            print( 'Nterminal episodes =', len(Nterminal_episodes),' of ', len(loop_stateL))
            progress_str = out_str
    print()
    if do_summ_print:
        s = ''
        if num_episodes >= max_num_episodes:
            s = '   (NOTE: STOPPED ON MAX-ITERATIONS)'

        print( 'Exited Epsilon Greedy, MC First-Visit Value Iterations', s )
        print( '   num episodes    =', num_episodes, ' (min limit=%i)'%min_num_episodes, ' (max limit=%i)'%max_num_episodes )
        print( '   gamma           =', gamma )
        print( '   estimated err   =', abserr )
        print( '   Error limit     =', max_abserr )
        print( 'Nterminal episodes =', len(Nterminal_episodes),' of ', len(loop_stateL))
    
        action_value_ave.summ_print(showRunningAve=showRunningAve, fmt_Q=fmt_Q )
        policy.summ_print(  environment=environment, verbosity=0, show_env_states=False  )
        
        try: # sims may not have a layout_print
            environment.layout_print( vname='reward', fmt=fmt_R, show_env_states=False, none_str='*')
        except:
            pass

    if save_pickle_file:
        policy.save_to_pickle_file( save_pickle_file )
        action_value_ave.save_to_pickle_file( save_pickle_file )
        
    return policy, action_value_ave

if __name__ == "__main__": # pragma: no cover
    
    from introrl.mdp_data.simple_grid_world import get_gridworld    
    gridworld = get_gridworld()
    
    
    policy, action_value = mc_epsilon_greedy( gridworld, initial_policy='default',
                                              first_visit=True, 
                                              do_summ_print=True, showRunningAve=False, 
                                              fmt_Q='%g', fmt_R='%g',
                                              max_num_episodes=1000, min_num_episodes=10, 
                                              max_abserr=0.001, gamma=0.9,
                                              iteration_prints=0)
                          
                          