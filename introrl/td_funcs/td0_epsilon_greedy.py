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
from introrl.agent_supt.state_value_coll import StateValueColl

from introrl.utils.banner import banner
from introrl.agent_supt.epsilon_calc import EpsilonGreedy
from introrl.agent_supt.alpha_calc import Alpha


def td0_epsilon_greedy( environment,  learn_tracker=None, # track progress of learning
                        initial_Vs=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                        read_pickle_file='', 
                        save_pickle_file='',
                        use_list_of_start_states=True, # use list OR single start state of environment.
                        do_summ_print=True, show_last_change=True, fmt_V='%g', fmt_R='%g',
                        pcent_progress_print=10,
                        show_banner = True,
                        max_num_episodes=1000, min_num_episodes=10, max_abserr=0.001, gamma=0.9,
                        iteration_prints=0,
                        max_episode_steps=10000,
                        epsilon=0.1, const_epsilon=True, epsilon_half_life=200,
                        alpha=0.1, const_alpha=True, alpha_half_life=200,
                        N_episodes_wo_decay=0):
    """
    ... GIVEN AN ENVIRONMENT ... 
    apply TD(0) Temporal Difference to find the OPTIMAL POLICY and STATE VALUES
    
    Returns: Policy and StateValueColl objects
    
    Use Episode Discounted Returns to find V(s), State-Value Function
    
    Terminates when abserr < max_abserr
    
    Assume that V(s), state_value_coll, has been initialized prior to call.
    
    Assume environment attached to policy will have method "get_any_action_state_hash"
    in order to begin at any action state.
    
    CREATES BOTH policy AND state_value_coll OBJECTS.
    """
    if show_banner:
        print('======= THIS IS AN EXPERIMENT TO CREATE TRANSITION PROBABILITIES w V(s) =========')
    
    eg = EpsilonGreedy(epsilon=epsilon, const_epsilon=const_epsilon, half_life=epsilon_half_life,
                       N_episodes_wo_decay=N_episodes_wo_decay)

    
    alpha_obj = Alpha( alpha=alpha, const_alpha=const_alpha, half_life=alpha_half_life )

    
    state_value_coll = StateValueColl( environment, init_val=initial_Vs )
    #state_value_coll.summ_print()
    num_s_hash = len( environment.get_all_action_state_hashes() )

    if read_pickle_file:
        policy.init_from_pickle_file( read_pickle_file )
        state_value_coll.init_from_pickle_file( read_pickle_file )
    
    if do_summ_print:

        print('================== EPSILON GREEDY DEFINED AS ========================')
        eg.summ_print()
        
        print('================== LEARNING RATE DEFINED AS ========================')
        alpha_obj.summ_print()
    
    if show_banner:
        s = 'Starting a Maximum of %i TD(0) Epsilon Greedy Episodes'%max_num_episodes +\
            '\nfor "%s" with Gamma = %g, Alpha = %g'%( environment.name, gamma, alpha_obj() )
        banner(s, banner_char='', leftMargin=0, just='center')
        
    # Iterate over a list of known possible start states
    if use_list_of_start_states:
        loop_stateL = environment.limited_start_state_list()
    else:
        #loop_stateL = [ random.choice( environment.limited_start_state_list() ) ]
        loop_stateL = [ environment.start_state_hash ]
        
    if show_banner:
        print('======================= Iterating over Start States ==================================')
        print( loop_stateL )
        print('======================================================================================')
    
    # set counter and flag
    episode_loop_counter = 0
    keep_looping = True
    
    #steps_per_episodeL = [] # track the number of steps in each episode
    #reward_sum_per_episodeL = [] # track sum of rewards during each episode
    
    progress_str = ''
    while (episode_loop_counter<=max_num_episodes-1) and keep_looping :
        
        keep_looping = False
        abserr = 0.0 # calculated below as part of termination criteria
        Nterminal_episodes = set() # tracks if start_hash got to terminal_set or max_num_episodes
        
        for start_hash in loop_stateL:
            episode_loop_counter += 1
            if episode_loop_counter > max_num_episodes:
                break
            
            if learn_tracker is not None:
                learn_tracker.add_new_episode()
            
            #reward_sum = 0.0
            s_hash = start_hash
            
            
            for n_episode_steps in range( max_episode_steps ):
                a_desc = state_value_coll.get_best_eps_greedy_action( s_hash, epsgreedy_obj=eg )
                #print('s_hash=%s'%str(s_hash), ' a_desc=%s'%str(a_desc))
                
                # Begin an episode
                if a_desc is None:
                    Nterminal_episodes.add( start_hash )
                    print('break for a_desc==None')
                    break
                else:
                    #print('s_hash=',s_hash,' a_desc=',a_desc)
                    sn_hash, reward = environment.get_action_snext_reward( s_hash, a_desc ) # prob-weighted choice
                    #reward_sum += reward
                    if learn_tracker is not None:
                        learn_tracker.add_sarsn_to_current_episode( s_hash, a_desc, reward, sn_hash)
                    
                    
                    if sn_hash is None:
                        Nterminal_episodes.add( start_hash )
                        print('break for sn_hash==None')
                        break
                    else:
            
                        state_value_coll.td0_update( s_hash=s_hash, a_desc=a_desc, 
                                                     alpha=alpha_obj(), gamma=gamma, 
                                                     sn_hash=sn_hash, reward=reward)
                        
                        if sn_hash in environment.terminal_set:
                            Nterminal_episodes.add( start_hash )
                            if (n_episode_steps==0) and (num_s_hash>2):
                                print('1st step break for sn_hash in terminal_set', sn_hash, 
                                      ' s_hash=%s'%str(s_hash), ' a_desc=%s'%str(a_desc))
                            break
                        s_hash = sn_hash
        
            # save the number of steps in each episode
            #steps_per_episodeL.append(n_episode_steps+1)
            #reward_sum_per_episodeL.append( reward_sum )
            
        
        # increment episode counter on EpsilonGreedy and Alpha objects
        eg.inc_N_episodes()
        alpha_obj.inc_N_episodes()
                
        abserr = state_value_coll.get_biggest_action_state_err()
        if abserr > max_abserr:
            keep_looping = True
            
        if episode_loop_counter < min_num_episodes:
            keep_looping = True # must loop for min_num_episodes at least
            
        pc_done = 100.0 * float(episode_loop_counter) / float(max_num_episodes)
        
        if pcent_progress_print > 0:
            out_str = '%3i%%'%( pcent_progress_print*(int(pc_done/float(pcent_progress_print)) ) )
        else:
            out_str = progress_str
        
        if out_str != progress_str:
            #score = environment.get_policy_score( policy=policy, start_state_hash=None, step_limit=1000)
            #print(out_str, ' score=%s'%str(score), ' = (r_sum, n_steps, msg)', end=' ')
            print( 'Nterminal episodes =', len(Nterminal_episodes),' of ', len(loop_stateL))
            progress_str = out_str
    #print()
    
    
    policy = Policy( environment=environment )
    for s_hash in environment.iter_all_action_states():
        a_desc = state_value_coll.get_best_eps_greedy_action( s_hash, epsgreedy_obj=None )
        policy.set_sole_action( s_hash, a_desc)
        
    
    if do_summ_print:
        s = ''
        if episode_loop_counter >= max_num_episodes:
            s = '   (NOTE: STOPPED ON MAX-ITERATIONS)'

        print( 'Exited Epsilon Greedy, TD(0) Value Iteration', s )
        print( '   # episodes      =', episode_loop_counter, ' (min limit=%i)'%min_num_episodes, ' (max limit=%i)'%max_num_episodes )
        print( '   gamma           =', gamma )
        print( '   estimated err   =', abserr )
        print( '   Error limit     =', max_abserr )
        print( 'Nterminal episodes =', len(Nterminal_episodes),' of ', len(loop_stateL))
    
        state_value_coll.summ_print(show_last_change=show_last_change, fmt_V=fmt_V )
        policy.summ_print(  environment=environment, verbosity=0, show_env_states=False  )
        
        try: # sims may not have a layout_print
            environment.layout_print( vname='reward', fmt=fmt_R, show_env_states=False, none_str='*')
        except:
            pass

        print('================== EPSILON GREEDY DEFINED AS ========================')
        eg.summ_print()

    if save_pickle_file:
        policy.save_to_pickle_file( save_pickle_file )
        state_value_coll.save_to_pickle_file( save_pickle_file )
        
    return policy, state_value_coll    #, steps_per_episodeL, reward_sum_per_episodeL

if __name__ == "__main__": # pragma: no cover
    
    from introrl.agent_supt.learning_tracker import LearnTracker    
    from introrl.mdp_data.simple_grid_world import get_gridworld    
    gridworld = get_gridworld()
    
    learn_tracker = LearnTracker()    
    
    #policy, action_value, steps_per_episodeL, reward_sum_per_episodeL = \
    policy, action_value = \
        td0_epsilon_greedy( gridworld,  learn_tracker=learn_tracker,
                            do_summ_print=True, show_last_change=True, 
                            fmt_V='%g', fmt_R='%g',
                            use_list_of_start_states=False, # use list OR single start state of environment.
                            max_num_episodes=1000, min_num_episodes=10, 
                            max_abserr=0.001, gamma=1.0,
                            alpha=0.1,    # const_alpha=False, alpha_half_life=1000,
                            epsilon=0.2, #  const_epsilon=False, epsilon_half_life=500,
                            max_episode_steps=1000,
                            iteration_prints=0)
                          
    
    learn_tracker.summ_print()
                              