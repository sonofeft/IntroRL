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
from introrl.utils.banner import banner
from introrl.agent_supt.epsilon_calc import EpsilonGreedy
from introrl.agent_supt.alpha_calc import Alpha
from introrl.policy import Policy
        
class SA_SemiGradAgent( object ):
    """
    SARSA or Qlearning semi-gradient agent for linear function approximator.
    """
    
    def __init__(self, environment,  learn_tracker=None, # track progress of learning
                  sa_linear_function=None, # if input, use it.
                  update_type='sarsa', # can be 'sarsa', 'qlearn'
                  read_pickle_file='', 
                  save_pickle_file='',
                  do_summ_print=True, show_last_change=True, 
                  pcent_progress_print=10,
                  show_banner = True,
                  gamma=0.9,
                  iteration_prints=0,
                  max_episode_steps=sys.maxsize,
                  epsilon=0.1, # can be constant or EpsilonGreedy object
                  alpha=0.1): # can be constant or Alpha object
        """
        ... GIVEN AN ENVIRONMENT ... 
        Use basic SARSA algorithm to solve for linear approximation of
        STATE-ACTION VALUES, Q(s,a)
        
        Each action is forced to be a DETERMINISTIC action leading to one state and reward.
        (If the next state or reward changes, only the new values will be considered)
            
        attribute: self.action_value_linfunc is the linear approximation, Q(s,a) object
        
        A DETERMINISTIC policy can be created externally from the self.action_value_linfunc attribute.
        """
        self.environment = environment
        self.learn_tracker = learn_tracker
        self.save_pickle_file = save_pickle_file
        
        self.do_summ_print = do_summ_print
        self.show_last_change = show_last_change
        self.pcent_progress_print = pcent_progress_print
        
        self.gamma = gamma
        self.iteration_prints = iteration_prints
        self.max_episode_steps = max_episode_steps
        
        self.num_episodes = 0
        self.num_updates = 0
        
        # if input epsilon is a float, use it to create an EpsilonGreedy object
        if type(epsilon) == type(0.1):
            self.epsilon_obj = EpsilonGreedy(epsilon=epsilon, const_epsilon=True)
        else:
            self.epsilon_obj = epsilon
        
        # if input alpha is a float, use it to create an Alpha object
        if type(alpha) == type(0.1):
            self.alpha_obj = Alpha(alpha=alpha, const_alpha=True)
        else:
            self.alpha_obj = alpha

        # create the action_value_linfunc for the environment.
        self.action_value_linfunc = sa_linear_function
        self.update_type = update_type
        
        if read_pickle_file:
            self.action_value_linfunc.init_from_pickle_file( read_pickle_file )
        
        if do_summ_print:
            print('================== EPSILON GREEDY DEFINED AS ========================')
            self.epsilon_obj.summ_print()
            
            print('================== LEARNING RATE DEFINED AS ========================')
            self.alpha_obj.summ_print()
        
        if show_banner:
            s = 'Starting a Maximum of %i SARSA Semi-Gradient Epsilon Greedy Steps/Episode'%self.max_episode_steps +\
                '\nfor "%s" with Gamma = %g, Alpha = %g'%( environment.name, self.gamma, self.alpha_obj() )
            banner(s, banner_char='', leftMargin=0, just='center')

    def run_episode(self, start_state, iter_sarsn=None):
        """
        Run a single episode of SARSA Semi-Gradient algorithm
        If iter_sarsn is input, use it instead of action_value_linfunc calculations.
        (Note: the start_state should NOT be in terminal_set if iter_sarsn is input.)
        """
        
        # increment episode counters
        self.num_episodes += 1
        self.epsilon_obj.inc_N_episodes()
        self.alpha_obj.inc_N_episodes()
        
        if self.learn_tracker is not None:
            self.learn_tracker.add_new_episode()
        
        # do SARSA Semi-Gradient loops until sn_hash in terminal_set
        s_hash = start_state
        
        n_steps_in_episode = 1
        while s_hash not in self.environment.terminal_set:
        
            if iter_sarsn is None:
                # get best epsilon-greedy action 
                a_desc = self.action_value_linfunc.get_best_eps_greedy_action( \
                                                s_hash, epsgreedy_obj=self.epsilon_obj )
                # check for bad action value
                if a_desc is None:
                    print('break for a_desc==None at s_hash=%s'%str(s_hash))
                    break
                
                # get next state and reward
                sn_hash, reward = self.environment.get_action_snext_reward( s_hash, a_desc )
            else:
                # retracing an existing episode
                s_hash, a_desc, reward, sn_hash = next( iter_sarsn )
                            
            if self.learn_tracker is not None:
                self.learn_tracker.add_sarsn_to_current_episode( s_hash, a_desc, reward, sn_hash)
            
            if sn_hash is None:
                print('break for sn_hash==None, #steps=',n_steps_in_episode,
                      ' s_hash=%s'%str(s_hash),' a_desc=%s'%str(a_desc))
                break
            
            # do RL update of Q(s,a) value
            if self.update_type == 'sarsa':
                an_desc = self.action_value_linfunc.get_best_eps_greedy_action( \
                                                sn_hash, epsgreedy_obj=self.epsilon_obj )
                
                self.action_value_linfunc.sarsa_update( s_hash=s_hash, a_desc=a_desc, 
                                                        alpha=self.alpha_obj(), gamma=self.gamma, 
                                                        sn_hash=sn_hash, an_desc=an_desc, 
                                                        reward=reward)
            elif self.update_type == 'qlearn':
                
                self.action_value_linfunc.qlearning_update( s_hash=s_hash, a_desc=a_desc, 
                                                            alpha=self.alpha_obj(), gamma=self.gamma, 
                                                            sn_hash=sn_hash, reward=reward)
                     
            self.num_updates += 1
            
            # keep a lid on the max number of episode steps.
            if n_steps_in_episode >= self.max_episode_steps:
                break
            
            # get ready for next loop
            n_steps_in_episode += 1
            s_hash = sn_hash

        #print(n_steps_in_episode, end=' ')


    def summ_print(self, long=True): # pragma: no cover
        """Show State objects in sorted state_hash order."""
        print('___ Policy Evaluation Agent Summary ___' )
        print('    Environment        = %s'%self.environment.name )
        print('    Update Type        = %s'%self.update_type )
        print('    Number of Episodes = %g'%self.num_episodes )
        
        print('================== EPSILON GREEDY FINAL ========================')
        self.epsilon_obj.summ_print()
        
        print('================== LEARNING RATE FINAL ========================')
        self.alpha_obj.summ_print()
    

if __name__ == "__main__": # pragma: no cover

    from introrl.mdp_data.simple_grid_world import get_gridworld
    from introrl.agent_supt.learning_tracker import LearnTracker
    from introrl.policy import Policy
    from introrl.linear_funcs.baseline_q_func import Baseline_Q_Func
    
    learn_tracker = LearnTracker()
    gridworld = get_gridworld( step_reward=-0.1 )
    #gridworld.summ_print(long=False)
    print('-'*77)    
    
    NUM_EPISODES = 2000
    
    alpha_obj = Alpha(alpha=0.1)
    alpha_obj.set_half_life_for_N_episodes( Nepisodes=NUM_EPISODES, alpha_final=0.03333333333333)
    
    eps_obj = EpsilonGreedy(epsilon=0.5)
    eps_obj.set_half_life_for_N_episodes( Nepisodes=NUM_EPISODES, epsilon_final=0.16666666666666)


    agent = SA_SemiGradAgent( environment=gridworld, 
                                sa_linear_function=Baseline_Q_Func( gridworld ),
                                learn_tracker=learn_tracker,
                                gamma=0.9,
                                alpha=alpha_obj,
                                epsilon=eps_obj)
    
    for i in range(NUM_EPISODES):
        agent.run_episode( (2,0))
    print()
    
    agent.summ_print()
    print('-'*77)
    #learn_tracker.summ_print()
    #print('-'*77)
    
    agent.action_value_linfunc.summ_print(fmt_Q='%.4f')
    print('-'*77)
    
    
    policy = Policy( environment=gridworld )
    for s_hash in gridworld.iter_all_action_states():
        a_desc = agent.action_value_linfunc.get_best_eps_greedy_action( s_hash, epsgreedy_obj=None )
        policy.set_sole_action( s_hash, a_desc)
    
    policy.summ_print( environment=gridworld, verbosity=0 )
    
    