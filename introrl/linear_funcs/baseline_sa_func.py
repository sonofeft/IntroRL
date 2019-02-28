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
import numpy as np
import random
import pickle
from introrl.agent_supt.change_tracker import ChangeTracker
from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.policy import Policy


class BaselineSAFunc( object ):
    """
    Create a linear function for an environment that simply one-hot encodes
    all of the state-action pairs.
    
    OVERRIDE THIS for more interesting linear functions.
    
    This is only interesting for debugging linear function solution routines.
    (i.e. each term in the one-hot encoding should move to near the actual 
    value function)
    """
    
    # ======================== OVERRIDE STARTING HERE ==========================
    def init_w_vector(self):
        """Initialize the weights vector and the number of entries, N."""
        
        # initialize a weights numpy array with random values.
        N = len(self.saD)
        self.w_vector = np.random.randn(N) / np.sqrt(N)
        self.N = len( self.w_vector )
                
    def get_sa_x_vector(self, s_hash, a_desc):
        """Return the x vector that represents the (s,a) pair."""
        x_vector = np.zeros(self.N, dtype=np.float)
        x_vector[ self.saD[(s_hash, a_desc)] ] = 1.0
        return x_vector
    # ======================== OVERRIDE ENDING HERE ==========================
    
    def __init__(self, environment):
        
        self.environment = environment
        
        # initialize known (s,a) pairs.
        self.saD = {}
        for s_hash in self.environment.iter_all_states():
            for a_desc in self.environment.get_state_legal_action_list( s_hash ):
                # set dict value to index of numpy array
                self.saD[ (s_hash, a_desc) ] = len(self.saD)
        
        # aD index=a_desc, value=last change to Q(s,a) value, float
        self.last_delta_QsaD = {} # index=s_hash value=aD (dict)
        
        self.chgTracker = ChangeTracker()
        self.init_tracking()
        
        # initialize a weights numpy array with random values.
        self.init_w_vector()
        # e.g.  self.w_vector = np.random.randn(self.N) / np.sqrt(self.N)
        self.N = len(self.w_vector)

    def init_tracking(self):
        # initialize to init_val for all states, terminal = 0.0
        for s_hash in self.environment.iter_all_states():
            if s_hash not in self.saD:
                self.saD[s_hash] = {}
                self.last_delta_QsaD[s_hash] = {}

            # may not be any actions in terminal state, so set None action.
            if s_hash in self.environment.terminal_set:
                self.last_delta_QsaD[s_hash][ a_desc ] = 0.0

            aL = self.environment.get_state_legal_action_list( s_hash )
            for a_desc in aL:
                self.last_delta_QsaD[s_hash][ a_desc ] = 0.0
                

    def get_number_of_changes(self):
        return self.chgTracker.get_number_of_changes()

    def num_Qsa(self):
        return len( self.saD )

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
                q = self.QsaEst( s_hash, a )
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

    def get_max_Qsa(self, s_hash):
        """return the maximum Q(s,a) for state, s_hash."""
        a_best = self.get_best_greedy_action( s_hash )
        if a_best is None:
            return None
        return self.QsaEst( s_hash, a_best )

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

    def QsaEst(self, s_hash, a_desc):
        """Return the current estimate for Q(s,a) from linear function eval."""
        
        x_vector = self.get_sa_x_vector( s_hash, a_desc )
        return self.w_vector.dot( x_vector )
    
    def get_gradient(self, s_hash, a_desc):
        """
        Return the gradient of value function with respect to w_vector.
        Since the function is linear in w, the gradient is = x_vector.
        """
        return self.get_sa_x_vector( s_hash, a_desc )

    def sarsa_update(self, s_hash='', a_desc='', alpha=0.1, gamma=1.0,
                     sn_hash='', an_desc='', reward=0.0):
        """
        Do a SARSA, Temporal-Difference-style learning rate update.
        Use estimated Q(s,a) values by evaluating linear function approximation.
        w = w + alpha * [R + gamma*QEst(s',a') - QEst(s,a)]
        """
        Qsat = self.QsaEst( s_hash, a_desc )
        
        if sn_hash in self.environment.terminal_set:
            delta = alpha * (reward - Qsat)
        else:
            Qsatp1 = self.QsaEst( sn_hash, an_desc )
            target_val = reward + gamma*Qsatp1

            delta = alpha * (target_val - Qsat)
        
        delta_vector = delta * self.get_gradient( s_hash, a_desc )
        self.w_vector += delta_vector

        # remember max amount of change due to [s_hash][a_desc]
        delta = np.max( np.absolute( delta_vector ) )
        self.record_changes( s_hash, a_desc, delta )

        return abs(delta) # return the absolute value of change

    def qlearning_update(self, s_hash='', a_desc='', sn_hash='',
                         alpha=0.1, gamma=1.0, reward=0.0):
        """
        Do a Q-Learning, Temporal-Difference-style learning rate update.
        Use estimated Q(s,a) values by evaluating linear function approximation.
        w = w + alpha * [R + gamma* max(QEst(s',a')) - QEst(s,a)]
        """
        Qsat = self.QsaEst( s_hash, a_desc )

        if sn_hash in self.environment.terminal_set:
            delta = alpha * (reward - Qsat)
        else:
            # find best Q(s',a')
            an_descL = self.environment.get_state_legal_action_list( sn_hash )
            
            if an_descL:
                best_a_desc, best_a_val = an_descL[0], float('-inf')
                for a in an_descL:
                    q = self.QsaEst( sn_hash, a )
                    if q > best_a_val:
                        best_a_desc, best_a_val = a, q
            else:
                best_a_val = 0.0
            
            # use best Q(s',a') to update Q(s,a)
            target_val = reward + gamma * best_a_val
            delta = alpha * (target_val - Qsat)

        delta_vector = delta * self.get_gradient( s_hash, a_desc )
        self.w_vector += delta_vector

        # remember max amount of change due to [s_hash][a_desc]
        delta = np.max( np.absolute( delta_vector ) )
        self.record_changes( s_hash, a_desc, delta )

        return abs(delta) # return the absolute value of change

    # ========================== pickle routines ===============================

    def make_pickle_filename(self, fname):
        """Make a file name ending with .qlf_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.qlf_pickle'

        else:
            fname = fname.replace(' ','_').replace('.','_') + '.qlf_pickle'

        return fname

    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        # build name for pickle
        fname = self.make_pickle_filename( fname )

        saveD = {}
        saveD['saD'] = self.saD
        saveD['last_delta_QsaD'] = self.last_delta_QsaD
        saveD['w_vector'] = self.w_vector

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

        saD = readD['saD']
        last_delta_QsaD = readD['last_delta_QsaD']
        w_vector = readD['w_vector']

        fileObject.close()
        print('Read ActionValueColl from file:',fname)

        return saD, last_delta_QsaD, w_vector

    def init_from_pickle_file(self, fname=None): # pragma: no cover
        """Initialize ActionValueColl from policy pickle file."""
        saD, last_delta_QsaD, w_vector = self.read_pickle_file( fname=fname )
        if saD:
            self.saD = saD
            self.w_vector = w_vector
            self.last_delta_QsaD = last_delta_QsaD
            self.N = len(self.w_vector)
            self.chgTracker = ChangeTracker()
            self.init_tracking()
        else:
            print('ERROR... Failed to read file:', fname)


    # ========================== summ_print ===============================

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
                    else: # s_hash is a legal state hash
                        aL = self.environment.get_state_legal_action_list( s_hash )
                        sL = [str(s_hash)]
                        ld_sL = [str(s_hash)]
                        for a_desc in aL:
                            qsa = self.QsaEst( s_hash, a_desc )
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
            for s_hash in self.environment.iter_all_states():
                aL = self.environment.get_state_legal_action_list( s_hash )
                for a_desc in aL:
                    qsa = self.QsaEst( s_hash, a_desc )
                
                    q = fmt_Q%qsa
                    s = '(%s, %s)='%(str(s_hash),str(a_desc)) + q.strip()
                    if show_last_change:
                        s = s + ' Last Delta = %s'%self.last_delta_QsaD[s_hash].get( a_desc, None)
                    
                    outL.append( s )
                    lmax_hash = max(lmax_hash, len(s))
            outL.sort() # sort in-place
            for s in outL:
                print('    ', s )

        

if __name__ == "__main__": # pragma: no cover
    import sys
    
    from introrl.mdp_data.simple_grid_world import get_gridworld
    
    gridworld = get_gridworld()

    oh = BaselineSAFunc( gridworld )
    
    SAVE_MODE = 0
    if SAVE_MODE:
        for i in range( oh.N ):
            oh.w_vector[i] = 0.1
    else:
        oh.init_from_pickle_file( fname='testing_lf_save')
    
    # -------------------------------
    old_w_vector = oh.w_vector.copy()
    
    oh.sarsa_update( s_hash=(0,0), a_desc='R', alpha=0.1, gamma=1.0,
                     sn_hash=(0,1), an_desc='R', reward=-1.0)
    
    oh.sarsa_update( s_hash=(0,2), a_desc='R', alpha=0.1, gamma=1.0,
                     sn_hash=(0,3), an_desc='R', reward=1.0)
    
    #oh.environment.layout = None
    oh.summ_print()
    sys.exit()
                     
    print('w_vector')
    for s_hash in gridworld.iter_all_states():
        for a_desc in gridworld.get_state_legal_action_list( s_hash ):
            i = oh.saD[ (s_hash, a_desc) ]
            if old_w_vector[i] == oh.w_vector[i]:
                print(s_hash, a_desc, '%.5f'%old_w_vector[i], '---> Both Equal')
            else:
                print(s_hash, a_desc, '%.5f'%old_w_vector[i], '%.5f'%oh.w_vector[i])
    
    if SAVE_MODE:
        oh.save_to_pickle_file( fname='testing_lf_save')
    
    print('='*66)
    print('oh.chgTracker:')
    oh.chgTracker.summ_print()
    print()
    print('Biggest a,s error =', oh.get_biggest_action_state_err() )
    
    
    
    sys.exit()
    print('Best Greedy Action at: (2,2)')
    print( oh.get_best_greedy_action( (2,2) ) )
    print('Max Q(s,a) at: (2,2)')
    print( oh.get_max_Qsa( (2,2) ) )
    
    for _ in range(10):
        oh.record_changes( (2,random.choice([0,1,2])), 'R', random.random() )
    print('oh.chgTracker:')
    oh.chgTracker.summ_print()
    print()
    print('Biggest a,s error =', oh.get_biggest_action_state_err() )
    print()
    print('='*66)
    
    print('x_vector')
    for s_hash in gridworld.iter_all_states():
        for a_desc in gridworld.get_state_legal_action_list( s_hash ):
            x_vector = oh.get_sa_x_vector( s_hash, a_desc )
            print(s_hash, a_desc, '[', ' '.join(['%g'%v for v in x_vector]), ']')
    print('='*66)
    print('qsa_est')
    for s_hash in gridworld.iter_all_states():
        for a_desc in gridworld.get_state_legal_action_list( s_hash ):
            qsa = oh.QsaEst( s_hash, a_desc )
            print(s_hash, a_desc, qsa)
    print('='*66)
    print('gradient')
    for s_hash in gridworld.iter_all_states():
        for a_desc in gridworld.get_state_legal_action_list( s_hash ):
            gradient = oh.get_gradient( s_hash, a_desc )
            print(s_hash, a_desc, '[', ' '.join(['%g'%v for v in gradient]), ']')
    