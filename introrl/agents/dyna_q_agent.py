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
from introrl.agent_supt.model import Model
from introrl.policy import Policy
from introrl.agent_supt.action_value_coll import ActionValueColl
        
class DynaQAgent( object ):
    """
    DynaQ Agent.
    """
    
    def __init__(self, environment,  learn_tracker=None, # track progress of learning
                  initial_Qsa=0.0, # init non-terminal_set of V(s) (terminal_set=0.0)
                  initial_action_value_coll=None, # if input, use it.
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
        Use basic Dyna-Q algorithm to solve for STATE-ACTION VALUES, Q(s,a)
        
        Each action is forced to be a DETERMINISTIC action leading to one state and reward.
        (If the next state or reward changes, only the new values will be considered)
            
        attribute: self.action_value_coll is the ActionValueColl, Q(s,a) object
        
        A DETERMINISTIC policy can be created externally from the self.action_value_coll attribute.
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

        # create the action_value_coll for the environment.
        if initial_action_value_coll is None:
            self.action_value_coll = ActionValueColl( environment, init_val=initial_Qsa )
        else:
            self.action_value_coll = initial_action_value_coll
        
        if read_pickle_file:
            self.action_value_coll.init_from_pickle_file( read_pickle_file )
        
        # initialize the model that will build from experience
        # do not build full model description on Model init, states not visited
        #  by the RL portion will have no returns values.
        self.model = Model( environment,  build_initial_model=False)
        #for s_hash, aD in self.action_value_coll.QsaD.items():
        #    for a_desc, Q in aD.items():
        #        self.model.add_action( s_hash, a_desc )
        
    
        if do_summ_print:
            print('================== EPSILON GREEDY DEFINED AS ========================')
            self.epsilon_obj.summ_print()
            
            print('================== LEARNING RATE DEFINED AS ========================')
            self.alpha_obj.summ_print()
        
        if show_banner:
            s = 'Starting a Maximum of %i Dyna-Q Epsilon Greedy Steps/Episode'%self.max_episode_steps +\
                '\nfor "%s" with Gamma = %g, Alpha = %g'%( environment.name, self.gamma, self.alpha_obj() )
            banner(s, banner_char='', leftMargin=0, just='center')

    def run_episode(self, start_state, Nplanning_loops=5, iter_sarsn=None):
        """
        Run a single episode of Dyna-Q algorithm
        If iter_sarsn is input, use it instead of action_value_coll calculations.
        (Note: the start_state should NOT be in terminal_set if iter_sarsn is input.)
        """
        
        # increment episode counters
        self.num_episodes += 1
        self.epsilon_obj.inc_N_episodes()
        self.alpha_obj.inc_N_episodes()
        
        if self.learn_tracker is not None:
            self.learn_tracker.add_new_episode()
        
        # do dyna_q loops until sn_hash in terminal_set
        s_hash = start_state
        
        n_steps_in_episode = 1
        while s_hash not in self.environment.terminal_set:
        
            if iter_sarsn is None:
                # get best epsilon-greedy action 
                a_desc = self.action_value_coll.get_best_eps_greedy_action( \
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
            
            #if s_hash in [(2,0),(4,0),'Gate_L']:
            #    print('%4i)'%n_steps_in_episode ,'  Now in',s_hash,'  Action=',a_desc,'  sn_hash=',sn_hash)
            
            # do RL update of Q(s,a) value
            self.action_value_coll.qlearning_update( s_hash=s_hash, a_desc=a_desc, sn_hash=sn_hash,
                                                     alpha=self.alpha_obj(), gamma=self.gamma, 
                                                     reward=reward)
            # give the above experience to the model
            self.model.add_action( s_hash, a_desc )
            
            # force DETERMINISTIC next state and reward.
            self.model.save_deterministic_action_results( s_hash, a_desc, sn_hash, reward_val=reward)
            
            # do NOT use simple save_action_results... it allows NON-DETERMINISTIC next state.
            #self.model.save_action_results( s_hash, a_desc, sn_hash, reward_val=reward)
            
            # --------------------------------- Planning Loop ------------------------
            # make Nplanning_loops calls to model
            for n_plan in range(Nplanning_loops):
                s_model = self.model.get_random_state()
                #print(s_model, end=' ')
                
                # vanilla DynaQ
                a_model = self.model.get_random_action( s_model )
                
                #sn_model, r_model = self.environment.get_action_snext_reward( s_model, a_model )
                sn_model, r_model = self.model.get_sample_sn_r( s_model, a_model)
                
                # update for the DynaQ  results.
                self.action_value_coll.qlearning_update( s_hash=s_model, a_desc=a_model, sn_hash=sn_model,
                                                         alpha=self.alpha_obj(), gamma=self.gamma, 
                                                         reward=r_model)
            
            
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
        print('    Number of Episodes = %g'%self.num_episodes )
        
    

if __name__ == "__main__": # pragma: no cover

    from introrl.mdp_data.sutton_dyna_grid import get_gridworld
    from introrl.agent_supt.learning_tracker import LearnTracker
    from introrl.policy import Policy
    
    learn_tracker = LearnTracker()
    gridworld = get_gridworld()
    #gridworld.summ_print(long=False)
    print('-'*77)    

    agent = DynaQAgent( environment=gridworld, 
                        learn_tracker=learn_tracker,
                        gamma=0.95)
    
    for i in range(20):
        print(i,end=' ')
        agent.run_episode( (2,0), Nplanning_loops=50)
    print()
    
    agent.summ_print()
    print('-'*77)
    learn_tracker.summ_print()
    print('-'*77)
    
    agent.action_value_coll.summ_print(fmt_Q='%.4f')
    print('-'*77)
    
    
    policy = Policy( environment=gridworld )
    for s_hash in gridworld.iter_all_action_states():
        a_desc = agent.action_value_coll.get_best_eps_greedy_action( s_hash, epsgreedy_obj=None )
        policy.set_sole_action( s_hash, a_desc)
    
    policy.summ_print( environment=gridworld, verbosity=0 )
    
    