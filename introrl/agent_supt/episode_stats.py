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
from introrl.utils.running_ave import RunningAve
        
class EpisodeStats( object ):
    """
    Keep a record of all states, actions, next_states and rewards
    experienced during a series of episodes.
    """
    
    def __init__(self, name='Episode Records', first_visit_type=None ):
        """ 
        Given a series of episodes, this will record a summary of them.
        first_visit_type can be None, 'S' for state or 'SA' for state-action
        
        When adding to stats, only add episode steps that conform to "first_visit_type"
        """
        self.name = name
        self.first_visit_type = first_visit_type
        
        self.action_stateD    = {} # index=s_hash, value=number of visits
        self.terminal_stateD = {} # index=s_hash, value=number of visits
        
        self.legal_actionsD  = {} # index=s_hash, value=set of legal a_desc
        self.taken_saD = {} # index=(s_hash, a_desc), value=number of times taken
        
        self.snext_rewardD = {} # index=(s_hash, a_desc), value=sn_rD
                                # sn_rD = {}   index=sn_hash, value=RunningAve of reward
    
    def store_info(self, s_hash, a_desc, reward, sn_hash ):
        
        # index=s_hash, value=number of visits
        self.action_stateD[ s_hash ] = self.action_stateD.get( s_hash, 0 ) + 1 
        
        # index=s_hash, value=set of legal a_desc
        if s_hash not in self.legal_actionsD:
            self.legal_actionsD[ s_hash ] = set()
        self.legal_actionsD[ s_hash ].add( a_desc ) 
        
        # index=(s_hash, a_desc), value=number of times taken
        self.taken_saD[(s_hash, a_desc)] = self.taken_saD.get( (s_hash, a_desc), 0) + 1 
        
        # index=(s_hash, a_desc), value=sn_rD
        #                               sn_rD = {}   index=sn_hash, value=RunningAve of reward
        if (s_hash, a_desc) not in self.snext_rewardD:
            self.snext_rewardD[ (s_hash, a_desc) ] = {} # sn_rD, index=sn_hash, value=RunningAve of reward
        
        if sn_hash not in self.snext_rewardD[ (s_hash, a_desc) ]:
            self.snext_rewardD[ (s_hash, a_desc) ][ sn_hash ] = RunningAve()
            
        self.snext_rewardD[ (s_hash, a_desc) ][ sn_hash ].add_val( reward ) 
                                
                                
    
    def add_episode(self, episode_obj):
        """Step through the episode, saving s_hash, a_desc, reward, sn_hash info."""
        
        visited_set = set() # if a first_visit_type, need temporary list
        
        for (s_hash, a_desc, reward, sn_hash) in episode_obj.iter_all_sars():
            
            if self.first_visit_type is None:
                self.store_info( s_hash, a_desc, reward, sn_hash )
                
            elif self.first_visit_type == 'S':
                if s_hash not in visited_set:
                    self.store_info( s_hash, a_desc, reward, sn_hash )
                visited_set.add( s_hash )
                
            elif self.first_visit_type == 'SA':
                if (s_hash, a_desc) not in visited_set:
                    self.store_info( s_hash, a_desc, reward, sn_hash )
                visited_set.add( (s_hash, a_desc) )
                
            else:
                raise ValueError( 'For First-Visit episodes, "first_visit_type" MUST be None, "S", or "SA"' )
        
        t_hash = episode_obj.terminal_state()
        if t_hash is not None:
            self.terminal_stateD[ t_hash ] = self.terminal_stateD.get( t_hash, 0) + 1 # index=s_hash, value=number of visits
        

    
    def summ_print(self):  # pragma: no cover
        print( 'Episode  Records:',self.name,'\nFirst Visit Type:', self.first_visit_type, end='\n\n')

        print( '    Visited States, Number of Visits, Action Set' )
        for s_hash, n in self.action_stateD.items():# index=s_hash, value=number of visits
            print('  %16s  %8i'%(s_hash, n),'        ', self.legal_actionsD[s_hash] )
            
        print( '    Terminal States, Number of Visits' )
        for s_hash, n in self.terminal_stateD.items():# index=s_hash, value=number of visits
            print('  %16s  %8i'%(s_hash, n) )

        print( '    State-Action Pairs,    Number of Times, (NextState, AveReward)' )
        for (s_hash, a_desc), n in self.taken_saD.items():# index=(s_hash, a_desc), value=number of times taken
            
            #self.snext_rewardD = {} # index=(s_hash, a_desc), value=sn_rD
            #             sn_rD = {}   index=sn_hash, value=RunningAve of reward
            snrL = [(sn_hash, R.get_ave()) for (sn_hash, R) in self.snext_rewardD[(s_hash, a_desc)].items()]
            
            print('    %16s %-10s  %8i'%(s_hash, a_desc, n),' ', snrL )
            
        

if __name__ == "__main__": # pragma: no cover
    
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    from introrl.policy import Policy
    from introrl.agent_supt.epsilon_calc import EpsilonGreedy
    from introrl.agent_supt.episode_maker import make_episode
    
    gridworld = get_gridworld()
    
    pi = Policy( environment=gridworld )
    
    pi.set_policy_from_piD( gridworld.get_default_policy_desc_dict() )
    #pi.summ_print()
    
    eg = EpsilonGreedy(epsilon=0.5, const_epsilon=True, half_life=200,
                   N_episodes_wo_decay=0)

    terminal_set = gridworld.get_set_of_all_terminal_state_hashes()

    episode = make_episode( (2,0), pi, gridworld, terminal_set, eps_greedy=eg )
    
    episode.summ_print()
    
    print('-'*55)
    ES = EpisodeStats()
    ES.add_episode( episode )
    ES.summ_print()    