#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
import os
import random
import pickle
import copy

from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.utils.running_ave import RunningAve
from introrl.agent_supt.change_tracker import ChangeTracker
from introrl.policy import Policy
from introrl.agent_supt.state_value_coll import StateValueColl

class ActionValueColl( object ):

    def __init__(self, environment, init_val=0.0):
        """
        A Collection of Action-Value, Q(s,a) floating point values
        for each state-action pair in the environment

        Each value can be updated with a learning rate (alpha)

        To get value use:
        qsa.get_val( s_hash, a_desc ):

        To update value use:
        sv.delta_update( s_hash, a_desc, delta)
        sv.sarsa_update( s_hash, a_desc, alpha, gamma,
                         sn_hash, an_desc, reward)

        (Terminal States have Q(s,a) = 0.0)
        """

        self.environment = environment

        self.QsaD = {} # index=s_hash value=aD (dict),  aD index=a_desc, value=Q(s,a) value, float
        
        # aD index=a_desc, value=last change to Q(s,a) value, float
        self.last_delta_QsaD = {} # index=s_hash value=aD (dict)
        
        self.chgTracker = ChangeTracker()
        
        self.init_Qsa_to_val( init_val )

    def get_number_of_changes(self):
        return self.chgTracker.get_number_of_changes()

    def merge_active_value_coll(self, av_coll_2):
        """Merge self and av_coll_2 into a single ActionValueColl object"""
        av_result = copy.deepcopy( self )
        for s_hash, aD in self.QsaD.items():
            for a_desc, Q in aD.items():
                av_result.QsaD[s_hash][a_desc] = (self.QsaD[s_hash][a_desc] +\
                                             av_coll_2.QsaD[s_hash][a_desc]) / 2.0
        return av_result

    def build_sv_from_av(self):
        """
        Build a StateValueColl from this ActionValueColl
        NOTE: Any policy derived directly from the resulting StateValueColl will 
        LIKELY BE DIFFERENT from a policy derived directly from this ActionValueColl.
        """
        
        sv = StateValueColl( self.environment )
        for s_hash, aD in self.QsaD.items():
            best_val = float('-inf')
            for a_desc, Q in aD.items():
                if self.QsaD[s_hash][a_desc] > best_val:
                    best_val = self.QsaD[s_hash][a_desc]
            sv.VsD[ s_hash ] = best_val
        
        return sv

    def num_Qsa(self):
        return len( self.QsaD )

    def init_Qsa_to_val(self, init_val):
        # initialize to init_val for all states, terminal = 0.0
        for s_hash in self.environment.iter_all_states():
            if s_hash not in self.QsaD:
                self.QsaD[s_hash] = {}
                self.last_delta_QsaD[s_hash] = {}

            # may not be any actions in terminal state, so set None action.
            if s_hash in self.environment.terminal_set:
                self.QsaD[s_hash][None] = 0.0
                self.last_delta_QsaD[s_hash][ a_desc ] = 0.0

            aL = self.environment.get_state_legal_action_list( s_hash )
            for a_desc in aL:
                self.last_delta_QsaD[s_hash][ a_desc ] = 0.0
                
                # some terminal states have actions to themselves.
                if s_hash in self.environment.terminal_set:
                    self.QsaD[s_hash][ a_desc ] = 0.0
                else:
                    self.QsaD[s_hash][ a_desc ] = init_val

    def get_best_eps_greedy_action(self, s_hash, epsgreedy_obj=None ):
        """
        Pick the best action for state "s_hash" based on max Q(s,a)
        If epsgreedy_obj is given, apply Epsilon Greedy logic to choice.
        """
        a_descL = self.environment.get_state_legal_action_list( s_hash )
        if a_descL:
            best_a_desc, best_a_val = a_descL[0], float('-inf')
            bestL = [best_a_desc]
            for a in a_descL:
                q = self.QsaD[s_hash][a]
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

    def get_best_greedy_action(self, s_hash):
        return self.get_best_eps_greedy_action( s_hash )

    def get_best_dbl_eps_greedy_action(self, av_coll_2, s_hash, epsgreedy_obj=None ):
        """
        Pick the best action for state "s_hash" based on COMBINED max Q(s,a)
        If epsgreedy_obj is given, apply Epsilon Greedy logic to choice.
        """
        a_descL = self.environment.get_state_legal_action_list( s_hash )
        if a_descL:
            best_a_desc, best_a_val = a_descL[0], float('-inf')
            bestL = [best_a_desc]
            for a in a_descL:
                q1 = self.QsaD[s_hash][a]
                q2 = av_coll_2.QsaD[s_hash][a]
                q = q1 + q2
                
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


    def record_changes(self, s_hash, a_desc, delta ):
        """Keep track of changes made to Q(s,a) values"""
        
        delta = abs(delta) # make sure that only absolute values are saved.
        
        # remove any record of last change to [s_hash][a_desc]
        self.chgTracker.dec_change( self.last_delta_QsaD[s_hash][ a_desc ] )
        
        # add delta to tracking record
        self.chgTracker.inc_change( delta )
        
        # remember that delta was last change to  [s_hash][a_desc]
        self.last_delta_QsaD[s_hash][ a_desc ] = delta
    
    def get_biggest_action_state_err(self):
        """Estimate the biggest error in all the action values."""
        #print('self.chgTracker.get_biggest_change()', self.chgTracker.get_biggest_change())
        return self.chgTracker.get_biggest_change()

    def get_val(self, s_hash, a_desc):
        """Return the  Action-Value for (s_hash, a_desc)"""
        return self.QsaD[s_hash][ a_desc ] # Allow key error

    def delta_update(self, s_hash='', a_desc='', delta=0.0):
        """
        Add delta to current value of Q(s,a) for s_hash, a_desc
        """
        self.QsaD[s_hash][a_desc] += delta

        # remember amount of change to [s_hash][a_desc]
        self.record_changes( s_hash, a_desc, delta )

    def qlearning_update(self, s_hash='', a_desc='', sn_hash='',
                         alpha=0.1, gamma=1.0, reward=0.0):
        """
        Do a Q-Learning, Temporal-Difference-style learning rate update.
        Q(s,a) = Q(s,a) + alpha * [R + gamma* max(Q(s',a')) - Q(s,a)]
        """
        Qsat = self.QsaD[s_hash][a_desc] # allow key error
        
        # find best Q(s',a')
        an_descL = self.environment.get_state_legal_action_list( sn_hash )
        
        if an_descL:
            best_a_desc, best_a_val = an_descL[0], float('-inf')
            for a in an_descL:
                q = self.QsaD[sn_hash][a]
                if q > best_a_val:
                    best_a_desc, best_a_val = a, q
        else:
            best_a_val = 0.0
        
        # use best Q(s',a') to update Q(s,a)
        target_val = reward + gamma * best_a_val
        delta = alpha * (target_val - Qsat)
        self.QsaD[s_hash][a_desc] += delta

        # remember amount of change to [s_hash][a_desc]
        self.record_changes( s_hash, a_desc, delta )

        return abs(delta) # return the absolute value of change

    def dbl_qlearning_update(self, av_coll_2, s_hash='', a_desc='', sn_hash='',
                         alpha=0.1, gamma=1.0, reward=0.0):
        """
        Do a Double Q-Learning, Temporal-Difference-style learning rate update.
        Given a 2nd ActionValueColl, av_coll_2, update EITHER self, or av_coll_2.
        
        Q(s,a) = Q(s,a) + alpha * [R + gamma* max(Q(s',a')) - Q(s,a)]
        """
        
        # randomly decide which Q(s,a) to update, self or av_coll_2
        if random.random() < 0.5:
            # use best Q(s',a') to update "self" Q(s,a)
            Qsat = self.QsaD[s_hash][a_desc] # allow key error
            best_a_desc = self.get_best_greedy_action( sn_hash )
            
            q = av_coll_2.QsaD[sn_hash][best_a_desc]
            target_val = reward + gamma * q
            delta = alpha * (target_val - Qsat)
            self.QsaD[s_hash][a_desc] += delta

            # remember amount of change to [s_hash][a_desc]
            self.record_changes( s_hash, a_desc, delta )
        else:
            # use best Q(s',a') to update "av_coll_2" Q(s,a)
            Qsat = av_coll_2.QsaD[s_hash][a_desc] # allow key error
            best_a_desc = av_coll_2.get_best_greedy_action( sn_hash )
            
            q = self.QsaD[sn_hash][best_a_desc]
            target_val = reward + gamma * q
            delta = alpha * (target_val - Qsat)
            av_coll_2.QsaD[s_hash][a_desc] += delta

            # remember amount of change to [s_hash][a_desc]
            av_coll_2.record_changes( s_hash, a_desc, delta )

        return abs(delta) # return the absolute value of change

    def sarsa_update(self, s_hash='', a_desc='', alpha=0.1, gamma=1.0,
                     sn_hash='', an_desc='', reward=0.0):
        """
        Do a SARSA, Temporal-Difference-style learning rate update.
        Q(s,a) = Q(s,a) + alpha * [R + gamma*Q(s',a') - Q(s,a)]
        """
        Qsat = self.QsaD[s_hash][a_desc] # allow key error
        Qsatp1 = self.QsaD[sn_hash][an_desc]
        target_val = reward + gamma*Qsatp1

        delta = alpha * (target_val - Qsat)
        self.QsaD[s_hash][a_desc] += delta

        # remember amount of change to [s_hash][a_desc]
        self.record_changes( s_hash, a_desc, delta )

        return abs(delta) # return the absolute value of change

    def expected_sarsa_update(self, s_hash='', a_desc='', 
                              alpha=0.1, gamma=1.0, epsilon=0.1,
                              sn_hash='', reward=0.0):
        """
        Do an Expected SARSA, Temporal-Difference-style learning rate update.
        Q(s,a) = Q(s,a) + alpha * [R + gamma * Expected[Q(s',a')] - Q(s,a)]
        """
        
        an_best = self.get_best_greedy_action( sn_hash )
        expected_val = (1.0-epsilon) * self.QsaD[sn_hash][an_best]

        an_descL = self.environment.get_state_legal_action_list( sn_hash )
        if an_descL:
            frac = epsilon / len(an_descL)
            for an_desc in an_descL:
                expected_val += frac * self.QsaD[sn_hash][an_desc]
                
        target_val = reward + gamma * expected_val

        delta = alpha * (target_val - self.QsaD[s_hash][a_desc])
        self.QsaD[s_hash][a_desc] += delta

        # remember amount of change to [s_hash][a_desc]
        self.record_changes( s_hash, a_desc, delta )

        return abs(delta) # return the absolute value of change
        
            

    def make_pickle_filename(self, fname):
        """Make a file name ending with .qsa_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.qsa_pickle'

        else:
            fname = fname.replace(' ','_').replace('.','_') + '.qsa_pickle'

        return fname

    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        # build name for pickle
        fname = self.make_pickle_filename( fname )

        saveD = {}
        saveD['QsaD'] = self.QsaD

        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
        print('Saved ActionValueColl to file:',fname)

    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle file."""

        fname = self.make_pickle_filename( fname )
        if not os.path.isfile( fname ):
            print('Pickle File NOT found:', fname)
            return False

        fileObject = open(fname,'rb')
        readD = pickle.load(fileObject)

        QsaD = readD['QsaD']

        fileObject.close()
        print('Read ActionValueColl from file:',fname)

        return QsaD

    def init_from_pickle_file(self, fname=None): # pragma: no cover
        """Initialize ActionValueColl from policy pickle file."""
        QsaD = self.read_pickle_file( fname=fname )
        if QsaD:
            self.QsaD = QsaD

    def get_max_last_delta_overall(self):
        """ get biggest entry in self.last_delta_QsaD # index=s_hash value=aD (dict)"""
        d_max = 0.0
        for aD in self.last_delta_QsaD.values():
            for val in aD.values():
                d_max = max(d_max, abs(val))
        return d_max

    def get_policy(self):
    
        policy = Policy( environment=self.environment )
        for s_hash in self.environment.iter_all_action_states():
            a_desc = self.get_best_greedy_action( s_hash )
            policy.set_sole_action( s_hash, a_desc)
        return policy

    def summ_print(self, fmt_Q='%.3f', none_str='*', show_states=True, 
                   show_last_change=True, show_policy=True):
        print()
        print('___ "%s" Action-Value Summary ___'%self.environment.name  )

        if self.environment.layout is not None:
            # make summ_print using environment.layout
            if show_states:
                self.environment.layout.s_hash_print( none_str='*' )

            row_tickL = self.environment.layout.row_tickL
            col_tickL = self.environment.layout.col_tickL
            x_axis_label = self.environment.layout.x_axis_label
            y_axis_label = self.environment.layout.y_axis_label

            d_max = self.get_max_last_delta_overall()
            if d_max==0.0:
                d_max = 1.0E-10

            rows_outL = []
            last_delta_rows_outL = [] # if show_last_change == True
            for row in self.environment.layout.s_hash_rowL:
                outL = []
                ld_outL = []
                for s_hash in row:
                    if not self.environment.is_legal_state( s_hash ):
                        if is_literal_str( s_hash ):
                            outL.append( s_hash[1:-1] )
                            ld_outL.append( s_hash[1:-1] )
                        else:
                            outL.append( none_str )
                            ld_outL.append( none_str )
                    else:
                        aD = self.QsaD[s_hash]
                        sL = [str(s_hash)]
                        ld_sL = [str(s_hash)]
                        for a_desc,qsa in aD.items():
                            s = fmt_Q%qsa
                            sL.append( '%s='%str(a_desc) + s.strip()  )
                            try:
                                d_val = int(100.0*self.last_delta_QsaD[s_hash].get( a_desc )/d_max)
                                if d_val > 0:
                                    lds = '%i%%'%d_val
                                    ld_sL.append( '%s='%str(a_desc) + lds.strip()  )
                                else:
                                    ld_sL.append( '%s~0'%str(a_desc) )
                            except:
                                ld_sL.append( '%s=None'%str(a_desc) )
                                
                        outL.append(  '\n'.join(sL).strip()  )
                        ld_outL.append(  '\n'.join(ld_sL).strip()  )
                rows_outL.append( outL )
                last_delta_rows_outL.append( ld_outL )

            print_string_rows( rows_outL, row_tickL=row_tickL, const_col_w=True,
                               line_chr='_', left_pad='    ', col_tickL=col_tickL,
                               header=self.environment.name + ' Action-Value Summary, Q(s,a)',
                               x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                               justify='right')

            if show_last_change:
                print_string_rows( last_delta_rows_outL, row_tickL=row_tickL, const_col_w=True,
                                   line_chr='_', left_pad='    ', col_tickL=col_tickL,
                                   header=self.environment.name + ' Last %% of Max Change to Q(s,a) Summary, (max change=%g)'%d_max,
                                   x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                                   justify='right')

            if show_policy:
                policy = self.get_policy()
                policy.summ_print(verbosity=0, environment=self.environment)

        # ------------------------- simple output w/o a layout ------------
        else:
            lmax_hash = 6

            outL = [] # list of strings "(s_hash,a_desc)=Q"
            for s_hash in self.QsaD.keys():
                for  a_desc,qsa in self.QsaD[s_hash].items():
                    q = fmt_Q%self.QsaD[s_hash][ a_desc ]
                    s = '(%s, %s)='%(str(s_hash),str(a_desc)) + q.strip()
                    if show_last_change:
                        s = s + ' Last Delta = %s'%self.last_delta_QsaD[s_hash].get( a_desc, None)
                    
                    outL.append( s )
                    lmax_hash = max(lmax_hash, len(s))
            outL.sort() # sort in-place
            for s in outL:
                print('    ', s )


if __name__ == "__main__": # pragma: no cover

    from introrl.policy import Policy
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()
    policyD = gridworld.get_default_policy_desc_dict()

    pi = Policy( environment=gridworld )
    #pi.learn_all_states_and_actions_from_env( gridworld )
    pi.set_policy_from_piD( policyD )

    # -------------

    av = ActionValueColl( gridworld )
    av.init_Qsa_to_val( 0.5 )

    print('Value at ((0,1),"R") is:', av.get_val( (0,1),"R" ) )
    av.sarsa_update( s_hash=(0,1), a_desc='R', alpha=0.1, gamma=1.0,
                     sn_hash=(0,2), an_desc='R', reward=1.0)

    #gridworld.layout = None

    av.summ_print( fmt_Q='%6g' )
    print('-'*55)
    gridworld.layout = None
    av.summ_print( fmt_Q='%6g' )

