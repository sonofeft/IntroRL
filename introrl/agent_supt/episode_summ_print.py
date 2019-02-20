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
from introrl.utils.grid_funcs import print_string_rows

def epi_summ_print(episode, policy, environment, show_rewards=False,
                   show_env_states=True, none_str='*'):
    
    """print the environment layout with the episode shown. """
    
    if environment.layout is None:
        print('...ERROR... "%s" tried to layout_print w/o a defined layout'%environment.name )
        return
        
    if show_env_states:
        environment.layout.s_hash_print( none_str=none_str )
    
    
    state_visitD = {} # index=s_hash, value=list of (N, action) OR (N, action, reward)
    
    Nvis = 1
    for (s_hash, a_desc, reward, sn_hash) in episode.iter_all_sars():
        if s_hash not in state_visitD:
            state_visitD[s_hash] = []
            
        if show_rewards:
            state_visitD[s_hash].append( '[%i->%s %g]'%(Nvis, str(a_desc), reward) )
        else:
            state_visitD[s_hash].append( '[%i->%s]'%(Nvis, str(a_desc)) )
            
        Nvis += 1
    
    if sn_hash not in state_visitD:
        state_visitD[sn_hash] = []
    state_visitD[sn_hash].append( 'T->'+str(sn_hash) )
    
            
    # start setting up output grid
    x_axis_label = environment.layout.x_axis_label
    y_axis_label = environment.layout.y_axis_label
    row_tickL =  environment.layout.row_tickL
    col_tickL =  environment.layout.col_tickL
    
    state_hash_set = set( environment.get_all_action_state_hashes() )
    
    rows_outL = []
    for row in environment.layout.s_hash_rowL:
        outL = []
        for s_hash in row:
            #if s_hash not in environment.SC.stateD:
            if s_hash not in state_hash_set:
                outL.append( none_str )
            else:
                val = state_visitD.get( s_hash, None )
                if val is None:
                    outL.append( none_str )
                else:
                    outL.append( '\n'.join(val) )
                                        
        rows_outL.append( outL )
    
    if rows_outL:
        print_string_rows( rows_outL,  const_col_w=True, 
                           line_chr='_', left_pad='    ', 
                           y_axis_label=y_axis_label, row_tickL=row_tickL, col_tickL=col_tickL,
                           header=environment.name + ' Episode Summary', 
                           x_axis_label=x_axis_label, justify='right')
    
    
    
if __name__ == "__main__":  # pragma: no cover
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    from introrl.policy import Policy
    from introrl.agent_supt.epsilon_calc import EpsilonGreedy
    from introrl.agent_supt.episode_maker import make_episode
    
    gridworld = get_gridworld()
    
    pi = Policy( environment=gridworld )
    
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    #pi.summ_print()
    
    eg = EpsilonGreedy(epsilon=0.2, const_epsilon=True, half_life=200,
                   N_episodes_wo_decay=0)

    
    episode = make_episode( (2,0), pi, gridworld, gridworld.terminal_set, eps_greedy=eg )
    
    episode.summ_print()
    
    epi_summ_print(episode, pi, gridworld, show_rewards=True,
                   show_env_states=True, none_str='*')
    
    epi_summ_print(episode, pi, gridworld, show_rewards=False,
                   show_env_states=True, none_str='*')
