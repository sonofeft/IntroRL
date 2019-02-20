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
from introrl.agent_supt.episode import Episode

def make_episode(start_state, policy, environment, terminal_set=None, 
                 episode=None, first_a_desc=None, # if input, use first action
                 max_steps=10000, eps_greedy=None):
    
        
    if terminal_set is None:
        terminal_set = environment.terminal_set
    
    if episode is None:
        episode = Episode( episode_name='Episode' )
    else:
        episode.clear()
    
    
    s_hash = start_state
    
    # if first action is dictated, use it.
    if first_a_desc is None:
        a_desc = policy.get_single_action( s_hash )
        #print('got single action =', a_desc)
    else:
        a_desc = first_a_desc
        #print('used input action =', a_desc)
    
    for _ in range( max_steps ):
        # break out of loop when terminal state reached.
        
        if a_desc is None:
            episode.set_done_flag()
            #print('a_desc is None for s_hash=',s_hash,' episode length =',len(episode))
            break            
            
        if eps_greedy is not None:
            legal_actionL = environment.get_state_legal_action_list( s_hash) #assumes incl_zero_prob=True
            a_desc = eps_greedy( a_desc, legal_actionL )
        
        #print('looking at s_hash, action =', s_hash, a_desc)
        sn_hash, reward = environment.get_action_snext_reward( s_hash, a_desc ) # prob-weighted choice
        
        if sn_hash is None:
            episode.set_done_flag()
            #print('sn_hash is None')
            break
        
        episode.add_step( s_hash, a_desc, reward, sn_hash)
        
        # maybe change to: if environment.is_terminal_state( sn_hash ):
        if (sn_hash in terminal_set):
            episode.set_done_flag()
            #print('sn_hash in terminal_set', sn_hash, terminal_set)
            break
        
        # get ready for next step
        s_hash = sn_hash
        
        a_desc = policy.get_single_action( s_hash )
        if a_desc is None:
            print('policy.get_single_action( "%s" ) ='%str(s_hash), a_desc,' episode length =',len(episode))
    
    # increment episode counter in EpsilonGreedy object.
    if eps_greedy is not None:
        eps_greedy.inc_N_episodes()
    
    return episode
    
if __name__ == "__main__":  # pragma: no cover
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    from introrl.policy import Policy
    from introrl.agent_supt.epsilon_calc import EpsilonGreedy
    from introrl.agent_supt.episode_stats import EpisodeStats
    
    gridworld = get_gridworld()
    
    pi = Policy( environment=gridworld )
    
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    #pi.summ_print()
    
    eg = EpsilonGreedy(epsilon=0.2, const_epsilon=True, half_life=200,
                   N_episodes_wo_decay=0)

    
    episode = make_episode( (2,0), pi, gridworld, gridworld.terminal_set, eps_greedy=eg )
    
    episode.summ_print()
    print('-'*55)
    print('  First "S" Visit Reversed Return Values.')
    rev_rtnL = episode.get_rev_discounted_returns( gamma=0.9, first_visit=True, visit_type='S')
    for _ in rev_rtnL:
        print( _ )
    # --------------------------------------------
    print('#'*55)
    episode = make_episode( (2,0), pi, gridworld, gridworld.terminal_set, eps_greedy=eg, episode=episode )
    
    episode.summ_print()
    print('-'*55)
    print('  First "S" Visit Reversed Return Values.')
    rev_rtnL = episode.get_rev_discounted_returns( gamma=0.9, first_visit=True, visit_type='S')
    for _ in rev_rtnL:
        print( _ )
    
    print('-'*55)
    ES = EpisodeStats()
    ES.add_episode( episode )
    ES.summ_print()