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
import numpy as np

class UpdateWVector( object ):
    """
    Update the w_vector of a FeatureFunction object (i.e. self.feature_func.w_vector).
    """
    
    def __init__(self, feature_func):
        
        self.feature_func = feature_func
        self.sim = feature_func.sim

    def get_best_eps_greedy_action(self, s_vector, epsgreedy_obj=None ):
        """
        Pick the best action for state "s_vector" based on max Q(s,a)
        If epsgreedy_obj is given, apply Epsilon Greedy logic to choice.
        """
        a_descL = self.sim.get_state_legal_action_list( s_vector )
        if a_descL:
            best_a_desc, best_a_val = a_descL[0], float('-inf')
            bestL = [best_a_desc]
            for a in a_descL:
                q = self.feature_func.get_QsaEst( a, s_vector )
                if q > best_a_val:
                    best_a_desc, best_a_val = a, q
                    bestL = [ a ]
                elif q == best_a_val:
                    bestL.append( a )
            
            best_a_desc = random.choice( bestL )
            if epsgreedy_obj is not None:
                best_a_desc = epsgreedy_obj( best_a_desc, a_descL )
                    
            return best_a_desc
        return None
        
    def get_best_greedy_action(self, s_vector):
        """Use eps greedy logic without an epsgreedy_obj."""
        return self.get_best_eps_greedy_action( s_vector )

    def get_max_Qsa(self, s_vector):
        """return the maximum Q(s,a) for state, s_vector."""
        a_best = self.get_best_greedy_action( s_vector )
        if a_best is None:
            return None, None
        return a_best, self.feature_func.get_QsaEst( a_best, s_vector )

    def sarsa_update(self, s_vector='', a_desc='', alpha=0.1, gamma=1.0,
                     sn_vector='', an_desc='', reward=0.0):
        """
        Do a SARSA, Temporal-Difference-style learning rate update.
        Use estimated Q(s,a) values by evaluating linear function approximation.
        w = w + alpha * [R + gamma*QEst(s',a') - QEst(s,a)] * grad(s,a)
        """
        Qsat = self.feature_func.get_QsaEst( a_desc, s_vector )
        
        if self.sim.is_terminal_state( s_vector=sn_vector ):
            delta = alpha * (reward - Qsat)
        else:
            Qsatp1 = self.feature_func.get_QsaEst( an_desc, sn_vector )
            target_val = reward + gamma*Qsatp1

            delta = alpha * (target_val - Qsat)
        
        delta_vector = delta * self.feature_func.get_gradient( a_desc, s_vector )
        self.feature_func.w_vector += delta_vector

        # remember max amount of change due to [s_vector][a_desc]
        #delta = np.max( np.absolute( delta_vector ) )
        #self.record_changes( s_vector, a_desc, delta )

        #return abs(delta) # return the absolute value of change

    def qlearning_update(self, s_vector='', a_desc='', sn_vector='',
                         alpha=0.1, gamma=1.0, reward=0.0):
        """
        Do a Q-Learning, Temporal-Difference-style learning rate update.
        Use estimated Q(s,a) values by evaluating linear function approximation.
        w = w + alpha * [R + gamma* max(QEst(s',a')) - QEst(s,a)] * grad(s,a)
        """
        Qsat = self.feature_func.get_QsaEst( a_desc, s_vector )

        if self.sim.is_terminal_state( s_vector=sn_vector ):
            delta = alpha * (reward - Qsat)
        else:
            # find best Q(s',a')
            an_descL = self.sim.get_state_legal_action_list( sn_vector )
            
            if an_descL:
                best_a_desc, best_a_val = an_descL[0], float('-inf')
                for a in an_descL:
                    q = self.feature_func.get_QsaEst( a, sn_vector )
                    if q > best_a_val:
                        best_a_desc, best_a_val = a, q
            else:
                best_a_val = 0.0
            
            # use best Q(s',a') to update Q(s,a)
            target_val = reward + gamma * best_a_val
            delta = alpha * (target_val - Qsat)

        delta_vector = delta * self.feature_func.get_gradient( a_desc, s_vector )
        self.feature_func.w_vector += delta_vector

        # remember max amount of change due to [s_vector][a_desc]
        #delta = np.max( np.absolute( delta_vector ) )
        #self.record_changes( s_vector, a_desc, delta )

        #return abs(delta) # return the absolute value of change


if __name__=="__main__":
    
    from introrl.continuous_sims.sim_continuous import ContinuousSimulation
    from introrl.continuous_sims.feature_func import FeatureFunction
    from introrl.agent_supt.epsilon_calc import EpsilonGreedy
    
    sim = ContinuousSimulation(name='Mountain Car', step_reward=-1.0)
        
    ff = FeatureFunction( sim, name='Proportional', init_w_val=None)
    
    uv = UpdateWVector( ff )
    
    s_vector = sim.get_s_vector()
    epsgreedy_obj = EpsilonGreedy(epsilon=0.5, 
                                  const_epsilon=True, half_life=200,
                                  N_episodes_wo_decay=0)
    
    
    a_best = uv.get_best_eps_greedy_action( s_vector, epsgreedy_obj=epsgreedy_obj )
    print('at',s_vector,'  a_best=',a_best)
    print('Best Qsa =', uv.get_max_Qsa(s_vector) )

