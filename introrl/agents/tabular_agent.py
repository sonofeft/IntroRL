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
import copy
from introrl.utils.banner import banner
from introrl.agent_supt.epsilon_calc import EpsilonGreedy
from introrl.agent_supt.alpha_calc import Alpha
from introrl.agent_supt.model import Model
from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.policy import Policy
from introrl.agent_supt.action_value_coll import ActionValueColl
from introrl.agent_supt.state_value_coll import StateValueColl
        
class TabularAgent( object ):
    
    def set_update_type(self, update_type='qlearn'):
        """To change update type after creating TabularAgent"""
        self.update_type = update_type

    def set_planning_type(self, planning_type='dynaq'):
        """To change planning type after creating TabularAgent"""
        self.planning_type = planning_type
    
    def __init__(self, environment,  
                 update_type='qlearn', # qlearn, sarsa, expsarsa, dblqlearn, td0
                 planning_type='dynaq',   # dynaq, dynaq+, prioritysweep
                 learn_tracker=None, # track progress of learning
                 initial_Qsa_Vs=0.0, # init non-terminal_set of Q(s,a) or V(s), (terminal_set=0.0)
                 initial_value_coll=None): # can be StateValueColl or ActionValueColl
        """
        ... GIVEN AN ENVIRONMENT ... 
        Use a variety of algorithms to solve for: 
        STATE-VALUES, V(s) or STATE-ACTION VALUES, Q(s,a)
            
        A policy can be created externally from the StateValueColl or ActionValueColl
        """
        
        self.environment = environment
        self.update_type = update_type
        
        self.planning_type = planning_type
        if planning_type is None:
            self.model = None
        else:
            self.model = Model( environment,  build_initial_model=False)
        
        # always build a learn_tracker... a production system might not.
        if learn_tracker is None:
            self.learn_tracker = LearnTracker()
        else:
            self.learn_tracker = learn_tracker
        
        self.value_coll = initial_value_coll
        if initial_value_coll is None:
            if update_type in ('qlearn', 'sarsa', 'expsarsa','dblqlearn'):
                self.value_coll = ActionValueColl( environment, init_val=initial_Qsa_Vs)
            elif update_type in ('td0'):
                self.value_coll = StateValueColl( environment, init_val=initial_Qsa_Vs )
            else:
                raise ValueError( 'update_type "%s" not recognized'%update_type )

        # for Double Q-Learn, need a second ActionValueColl
        if update_type == 'dblqlearn':
            self.av_coll_2 = copy.deepcopy( self.value_coll )


        # make an EpsilonGreedy object for later use.
        self.epsilon_obj = EpsilonGreedy( epsilon=0.1, const_epsilon=True, half_life=200,
                                          N_episodes_wo_decay=0, greedy_mode=False)
        
        self.num_steps = 0    # duplicates what's in learn_tracker (if provided)
        self.num_episodes = 0 # duplicates what's in learn_tracker (if provided)
    

    def make_update_step(self, start_state, sarsn=None,
                         gamma=0.9, epsilon=0.1, alpha=0.1):
    
        """
        Make an update to StateValueColl or ActionValueColl based on update_type.
        If sarsn is input (s_hash, a_desc, reward, sn_hash), then just do it.
        
        Do an RL update
        """
        if start_state in self.environment.terminal_set:
            print('WARNING... called make_update_step with a terminal state')
            return
        
        self.num_steps += 1

        # set epsilon... it could get used a number of places.
        self.epsilon_obj.set_const_epsilon( epsilon_inp=epsilon )

        # discover the action, reward and next state 
        if sarsn is None:
            s_hash = start_state
            
            a_desc = self.value_coll.get_best_eps_greedy_action( \
                                            s_hash, epsgreedy_obj=self.epsilon_obj )
            
            if a_desc is None:
                print('break in make_update_step for a_desc==None at s_hash=%s'%str(s_hash))
                return
                
            # get next state and reward
            sn_hash, reward = self.environment.get_action_snext_reward( s_hash, a_desc )
            
        else:
            s_hash,a_desc,reward,sn_hash = sarsn
        
        # now that (s_hash, a_desc, reward, sn_hash) is known, save it to learn_tracker.
        if self.learn_tracker is not None:
            self.learn_tracker.add_sarsn_to_current_episode( s_hash, a_desc, reward, sn_hash)
        
        # check for bad sn_hash
        if sn_hash is None:
            print('break in make_update_step for sn_hash==None, #steps=',n_steps_in_episode,
                  ' s_hash=%s'%str(s_hash),' a_desc=%s'%str(a_desc))
            return
        
        # ------------------- do the appropriate update --------------------------------        
        self.do_a_value_update( s_hash, a_desc, reward, sn_hash, 
                                alpha=alpha, gamma=gamma, epsilon=epsilon)
                
        # -------------------------------------- save to planning model --------------------------
        # give the above experience to the model
        if self.model is not None:
            self.model.add_action( s_hash, a_desc )
            self.model.save_action_results( s_hash, a_desc, sn_hash, reward_val=reward,
                                            time_stamp=self.num_steps)
        
        return sn_hash # next step in episode may need next state description.
    
    def make_a_planning_update(self, gamma=0.9, epsilon=0.1, alpha=0.1):
        """
        Use model data to do some planning... (i.e. update value_coll based on experience)
        """
        s_hash = self.model.get_random_state()
        a_desc = self.model.get_random_action( s_hash )
        sn_hash, reward = self.model.get_sample_sn_r( s_hash, a_desc)
        
        self.do_a_value_update( s_hash, a_desc, reward, sn_hash, model_update=True,
                                alpha=alpha, gamma=gamma, epsilon=epsilon)

    def do_a_value_update(self, s_hash, a_desc, reward, sn_hash,  model_update=False,
                          alpha=0.1, gamma=0.9, epsilon=0.1):
        """Used for both RL and model updates."""
        
        if (s_hash is None) or (a_desc is None):
            return

        # ------------------- do the appropriate update --------------------------------
        if self.update_type=='qlearn':
        # do RL update of Q(s,a) value
            self.value_coll.qlearning_update( s_hash=s_hash, a_desc=a_desc, sn_hash=sn_hash,
                                                     alpha=alpha, gamma=gamma, 
                                                     reward=reward)
        elif self.update_type=='sarsa':
            
            # use model data if possible for next action.
            if model_update:
                an_desc = self.model.get_random_action( sn_hash )
                
            if not model_update or (an_desc is None):
                self.epsilon_obj.set_const_epsilon( epsilon_inp=epsilon )
                an_desc = self.value_coll.get_best_eps_greedy_action( sn_hash, 
                                                                      epsgreedy_obj=self.epsilon_obj )
            #print( s_hash, a_desc, reward, sn_hash )
            self.value_coll.sarsa_update( s_hash=s_hash, a_desc=a_desc, 
                                                 alpha=alpha, gamma=gamma, 
                                                 sn_hash=sn_hash, an_desc=an_desc, 
                                                 reward=reward)
        elif self.update_type=='expsarsa':

            self.value_coll.expected_sarsa_update( s_hash=s_hash, a_desc=a_desc, 
                                                   alpha=alpha, gamma=gamma, 
                                                   epsilon=epsilon,
                                                   sn_hash=sn_hash,
                                                   reward=reward)
        elif self.update_type == 'dblqlearn':
            self.value_coll.dbl_qlearning_update(self.av_coll_2, 
                                                 s_hash=s_hash, a_desc=a_desc, 
                                                 sn_hash=sn_hash,
                                                 alpha=alpha, gamma=gamma, 
                                                 reward=reward)
    
        elif self.update_type == 'td0':
            self.value_coll.td0_update( s_hash=s_hash, a_desc=a_desc, 
                                        alpha=alpha, gamma=gamma, 
                                        sn_hash=sn_hash, reward=reward)
            

    def run_episode(self, start_state, Nplanning_loops=5, iter_sarsn=None,
                    gamma=0.9, epsilon=0.1, alpha=0.1, 
                    max_episode_steps=sys.maxsize): # max steps set to virtual infinity here.
        """
        Run a single episode from start_state to a terminal state
        If iter_sarsn is input, use it instead of value_coll calculations.
        
        (Note: the start_state should NOT be in terminal_set if iter_sarsn is input.)
        """
        
        # increment episode counter
        self.num_episodes += 1
        
        if self.learn_tracker is not None:
            self.learn_tracker.add_new_episode()

        # set epsilon... it could get used a number of places.
        self.epsilon_obj.set_const_epsilon( epsilon_inp=epsilon )

        # call make_update_step until sn_hash in terminal_set
        s_hash = start_state
        
        n_steps_in_episode = 1
        while s_hash not in self.environment.terminal_set:
        
            if iter_sarsn is None:
                # Use value_coll to decide on best next action.
                sn_hash = self.make_update_step( s_hash, sarsn=None, 
                                                 gamma=gamma, epsilon=epsilon, alpha=alpha)
            else:
                # retracing an existing episode
                s_hash, a_desc, reward, sn_hash = next( iter_sarsn )
                sn_hash = self.make_update_step( s_hash, sarsn=(s_hash, a_desc, reward, sn_hash), 
                                                 gamma=gamma, epsilon=epsilon, alpha=alpha)
            
            # -------------------------- Planning Loop ------------------------
            # make Nplanning_loops calls to model
            for n_plan in range(Nplanning_loops):
                self.make_a_planning_update(gamma=gamma, epsilon=epsilon, alpha=alpha)
            
            
            # keep a lid on the max number of episode steps.
            if n_steps_in_episode >= max_episode_steps:
                break
            
            # get ready for next loop
            n_steps_in_episode += 1
            s_hash = sn_hash



if __name__ == "__main__": # pragma: no cover

    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    
    TA = TabularAgent( gridworld,  
                 update_type='qlearn', # qlearn, sarsa, expsarsa, dblqlearn, td0
                 planning_type='dynaq',   # dynaq, dynaq+, prioritysweep
                 learn_tracker=None, # track progress of learning
                 initial_Qsa_Vs=0.0, # init non-terminal_set of Q(s,a) or V(s), (terminal_set=0.0)
                 initial_value_coll=None) # can be StateValueColl or ActionValueColl

        
    TA.make_update_step( gridworld.start_state_hash, sarsn=None,
                         gamma=0.9, epsilon=0.1, alpha=0.1)
    
    for i in range(20):
        TA.run_episode(gridworld.start_state_hash, Nplanning_loops=10, iter_sarsn=None,
                        gamma=0.9, epsilon=0.1, alpha=0.2, 
                        max_episode_steps=sys.maxsize)
    
    TA.model.summ_print(long=False, time_stamp=TA.num_steps)
    TA.value_coll.summ_print()
    