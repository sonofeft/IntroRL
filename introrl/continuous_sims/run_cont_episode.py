#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import sys
import random
from introrl.agent_supt.episode import Episode

def run_episode(start_state, updater, update_type='sarsa', # can be 'sarsa', 'qlearn'
                 episode=None, first_a_desc=None, # if input, use first action
                 max_steps=10000, epsgreedy_obj=None, alpha=0.1, gamma=1.0,
                 plot_update_func=None): # plot_update_func can update an active plot
    
        
    if episode is None:
        episode = Episode( episode_name='Episode' )
    else:
        episode.clear()
    
    
    s_vector = start_state
    
    # if first action is dictated, use it.
    if first_a_desc is None:
        a_desc = updater.get_best_eps_greedy_action( s_vector, epsgreedy_obj=epsgreedy_obj )
        #print('got single action =', a_desc)
    else:
        a_desc = first_a_desc
        #print('used input action =', a_desc)
    
    for _ in range( max_steps ):
        # break out of loop when terminal state reached.
        
        if a_desc is None:
            episode.set_done_flag()
            #print('a_desc is None for s_vector=',s_vector,' episode length =',len(episode))
            break            
            
        if epsgreedy_obj is not None:
            legal_actionL = updater.sim.get_state_legal_action_list( s_vector )
            a_desc = epsgreedy_obj( a_desc, legal_actionL )
        
        #print('looking at s_vector, action =', s_vector, a_desc)
        sn_vector, reward = updater.sim.get_action_snext_reward( a_desc, s_vector )
        
        if sn_vector is None:
            episode.set_done_flag()
            #print('sn_vector is None')
            break
        
        episode.add_step( s_vector, a_desc, reward, sn_vector)
        
        if plot_update_func is not None:
            plot_update_func( s_vector, a_desc, reward, sn_vector )
        
        if update_type=='sarsa':
            # get next action assuming greedy policy
            an_desc = updater.get_best_greedy_action( sn_vector )
            updater.sarsa_update( s_vector=s_vector, a_desc=a_desc, alpha=alpha, gamma=gamma,
                                  sn_vector=sn_vector, an_desc=an_desc, reward=reward)
        elif update_type=='qlearn':
            updater.qlearning_update( s_vector=s_vector, a_desc=a_desc, sn_vector=sn_vector,
                                      alpha=alpha, gamma=gamma, reward=reward)
        else:
            raise ValueError('update_type=%s.  MUST be sarsa or qlearn.'%update_type)
        
        if updater.sim.is_terminal_state( sn_vector ):
            episode.set_done_flag()
            #print('sn_vector is terminal', sn_vector)
            break
        
        # get ready for next step
        s_vector = sn_vector
        
        # get next action for policy
        a_desc = updater.get_best_eps_greedy_action( s_vector, epsgreedy_obj=epsgreedy_obj )
        if a_desc is None:
            print('policy.get_single_action( "%s" ) ='%str(s_vector), a_desc,' episode length =',len(episode))
    
    
    return episode
    
if __name__ == "__main__":  # pragma: no cover
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    
    from introrl.continuous_sims.mountain_car import MountainCar
    from introrl.continuous_sims.feature_func import FeatureFunction
    from introrl.continuous_sims.feature_func_polynomial import FFPolynomial
    from introrl.continuous_sims.feat_func_tiles import FeatFuncTiles
    from introrl.continuous_sims.update_w_vector import UpdateWVector
    
    from introrl.agent_supt.epsilon_calc import EpsilonGreedy
    from introrl.agent_supt.alpha_calc import Alpha
    from introrl.continuous_sims.plot_feature_func import plot_policy, plot_cost_to_go
    
    sim = MountainCar(name='Mountain Car', step_reward=-1.0)
        
    ff = FeatureFunction( sim, name='Proportional', init_w_val=0.0)
    #ff = FFPolynomial(sim, name='Polynomial', init_w_val=0.0, n_degree=3, interaction_only=False)
    #ff = FeatFuncTiles(sim, name='TilingsInf', init_w_val=None, num_tiles=8, recenter=True, num_regionsL=[8,8,8])
        
    ff.init_from_pickle_file( 'mcar_' + ff.desc() )
        
    updater = UpdateWVector( ff )
    
    NUM_EPISODES = 50
    UPDATE_TYPE = 'sarsa'
    UPDATE_TYPE = 'qlearn'
    
    GAMMA = 1.0
    #GAMMA = 0.99
    
    eg = EpsilonGreedy(epsilon=0.05, const_epsilon=True, half_life=200, N_episodes_wo_decay=0)
    eg.set_half_life_for_N_episodes( Nepisodes=NUM_EPISODES, epsilon_final=0.00001)
    eg = None
    
    ALPHA = 0.05
    if hasattr(ff, 'num_tiles'):
        alpha_obj = Alpha( alpha=ALPHA/ff.num_tiles )
    else:
        alpha_obj = Alpha( alpha=ALPHA )
    
    #alpha_obj.set_half_life_for_N_episodes( Nepisodes=NUM_EPISODES, alpha_final=0.001)
    
    ep_lenL = []
    for loop_counter in range(NUM_EPISODES):
        
        if loop_counter%100==0:
            print()
            print('Loop:',loop_counter)
            sys.stdout.flush()
            
        sim.reset()
        x_init = -0.6 + 0.2*random.random()
        #x_init = -0.5
        
        episode = run_episode( (x_init,0), updater, update_type=UPDATE_TYPE, 
                               epsgreedy_obj=eg, max_steps=10000, gamma=GAMMA, alpha=alpha_obj() )
        print(len(episode), end=' ')
        sys.stdout.flush()
        ep_lenL.append( len(episode) )
        
        alpha_obj.inc_N_episodes()
        if eg is not None:
            eg.inc_N_episodes()
            

    # make final greedy run
    sim.reset()
    
    x_init = -0.5    
    
    episode = run_episode( (x_init,0), updater, update_type=UPDATE_TYPE, 
                           epsgreedy_obj=None, max_steps=10000, gamma=GAMMA, alpha=alpha_obj() )
    print('\nLAST GREEDY RUN')
    print(len(episode), end=' ')
    ep_lenL.append( len(episode) )


    #episode.summ_print()
    print('-'*55)
        
    if 0:
        fig, ax = plt.subplots()
        ax.plot( ep_lenL )
        ax.set(title='%s w FeatureFunction=%s\nUpdate Type = %s'%(sim.name, ff.name, UPDATE_TYPE.upper()) )
        
        plt.ylabel('Steps per Episode')
        plt.xlabel('Episode Number')
        
        #plt.show()
        plot_policy( updater, Ngrid=100, do_show=True )
        #plot_cost_to_go( updater, Ngrid=100, do_show=True )
    
    ff.save_to_pickle_file( 'mcar_' + ff.desc() )
    