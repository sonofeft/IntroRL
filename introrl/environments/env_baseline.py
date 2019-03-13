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
#import sys
import math
import pickle
import random

from introrl.layouts.generic_layout import GenericLayout
from introrl.transition_coll import TransitionColl
from introrl.reward import Reward, CONST, TABLE, FUNC
from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.utils.running_ave import RunningAve
from introrl.reward import Reward
from introrl.state import State
from introrl.utils.banner import banner

from introrl.environments.define_state_moves import DefineStateMoves

here = os.path.abspath(os.path.dirname(__file__))
up_one = os.path.split( here )[0] 

USER_HOME_DIR = os.path.dirname( os.path.expanduser('~/') )
mdp_path = os.path.join( USER_HOME_DIR, 'IntroRL_MDP' )
#mdp_path = os.path.join( up_one, 'mdp_data' )

class EnvBaseline( object ):


    def __init__(self, name='Generic Environment', s_hash_rowL=None, 
                 row_tickL=None, x_axis_label='',
                 col_tickL=None, y_axis_label='', 
                 colorD=None, basic_color='',
                 mdp_file=None):
        """
        A Basic Environment from which all others derive.
        If "s_hash_rowL" is input, it will be used to calc state layout
        
        colorD and basic_color affect the GenericLayout when saved as an image.
        """
        
        self.name = name
        
        self.define_statesD = {} # index=s_hash: value=DefineStateMoves object for s_hash
        
        self.TC = TransitionColl( name=name + ' TransitionColl' )
        
        
        self.default_policyD = None # may define later.

        # for convenience, make TransitionColl objects available locally
        self.SAC = self.TC.sa_coll     # share sa_coll with TransitionColl
        self.AC  = self.TC.action_coll # share action_coll with TransitionColl
        self.SC  = self.TC.state_coll  # share state_coll with TransitionColl
        
        self.define_environment()
        
        # can define a start state list smaller than all action states
        # (to create it, call define_limited_start_state_list( state_list )
        self.defined_limited_start_state_list = None
        
        self.terminal_set, self.action_state_set = self.TC.get_terminal_set_and_action_set()
        
        self.info = """A Basic Environment for solving Reinforcement Learning Problems."""

        self.layout = GenericLayout( self, s_hash_rowL=s_hash_rowL,
                                     row_tickL=row_tickL, x_axis_label=x_axis_label,
                                     col_tickL=col_tickL, y_axis_label=y_axis_label,
                                     colorD=colorD, basic_color=basic_color)
        
        self.failed_mdp_file_read = False
        if mdp_file is not None:
            if not self.read_pickle_file( mdp_file ):
                print('WARNING...   FAILED TO OPEN MDP FILE:', mdp_file)
                print('='*66)
                print('='*66)
                print('='*66)
                #sys.exit()
                self.failed_mdp_file_read = True
    
    def get_policy_score(self, policy=None, start_state_hash=None, step_limit=1000):
        """
        Given a Policy object, OR policy dictionary,
        apply it to the Environment and return a score
        
        Can iterate over limited_start_state_list, or simply start at start_state_hash.
        """
        
        if policy is None:
            policy = self.SAC
        
        if start_state_hash is None:
            s_hash = self.start_state_hash
        else:
            s_hash = start_state_hash
            
        r_sum = 0.0
        n_steps = 0
        a_desc = policy.get( s_hash, None)
        #print( policy )
        
        while (a_desc is not None) and (n_steps<step_limit):
            
            sn_hash, reward = self.get_action_snext_reward( s_hash, a_desc )
            
            try: # if reward is numeric, add to r_sum
                r_sum += reward
            except:
                pass
                
            n_steps += 1
            
            s_hash = sn_hash
            a_desc = policy.get( s_hash, None)
        
            
        msg = '' # any special message(s)
        return (r_sum, n_steps, msg)# can OVERRIDE this to return a more meaningful score.
    
    def make_pickle_filename(self, fname):
        """Make a file name ending with .mdp_pickle """
        if fname is None:
            fname = self.name.replace(' ','_').replace('.','_') + '.mdp_pickle'
            
        else:
            fname = fname.replace(' ','_').replace('.','_') + '.mdp_pickle'
            
        return fname
    
    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        
        #raise ValueError( 'save_to_pickle_file is BROKEN... DO NOT USE' )
        
        fname = self.make_pickle_filename( fname )
            
        print('Saving Environment to pickle file:', fname)
        saveD = {}
        saveD['name'] = self.name
        saveD['define_statesD'] = self.define_statesD
        saveD['info'] = self.info
        saveD['layout'] = self.layout
        
        if hasattr(self,'start_state_hash'):
            saveD['start_state_hash'] = self.start_state_hash
        
        if hasattr(self, 'defined_limited_start_state_list'):
            saveD['defined_limited_start_state_list'] = self.defined_limited_start_state_list
        
        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
    
    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle"""
        
        #raise ValueError( 'read_pickle_file is BROKEN... DO NOT USE' )
        
        
        fname = self.make_pickle_filename( fname )
        if os.path.isfile( fname ):
            pass # all good
        elif os.path.isfile( os.path.join( mdp_path, fname ) ):
            fname = os.path.join( mdp_path, fname )
        else:
            print('Pickle File NOT found:', fname)
            print('mdp_path:',mdp_path)
            
            s = '''Try running: "introrl_build_mdp" to create MDP Pickle Files.
Type: introrl_build_mdp
at the command line.'''
            banner(s, banner_char='', leftMargin=0, just='center')
            
            
            return False
        
        fileObject = open(fname,'rb')  
        
        readD = pickle.load(fileObject)  
        
        self.name = readD['name']
        self.define_statesD = readD['define_statesD']
        self.info = readD['info']
        self.layout = readD['layout']
        
        if 'start_state_hash' in readD:
            self.start_state_hash = readD['start_state_hash']
        if 'defined_limited_start_state_list' in readD:
            self.defined_limited_start_state_list = readD['defined_limited_start_state_list']
            
        self.define_env_states_actions() # use define_statesD to initialize data structures
        # ----------------------
        
        fileObject.close()
        
        return True
        
    def set_info(self, info):
        """Input string that describes Environment."""
        self.info = info
    
    def get_info(self):
        lmax = max( [len(s) for s in self.info.split('\n')] )
        lmax = max( 16, lmax )
        
        return '\n' + 'INFO'.center(lmax, '_') + '\n' + self.info + '\n' + '_'*lmax + '\n'
    
    def add_action_dict(self, actionD):
        """
        iterate through dictionary of actions calling "add_action" for each one.
        actionD, index=s_hash, value=list of a_desc
        """
        for s_hash, aL in actionD.items():
            a_prob = 1.0 / float(len(aL))
            for a_desc in aL:
                self.add_action(s_hash, a_desc, a_prob=a_prob)
    
    def add_action(self, s_hash, a_desc, a_prob=1.0):
        if s_hash not in self.define_statesD:
            self.define_statesD[ s_hash ] = DefineStateMoves( s_hash )
        
        self.define_statesD[ s_hash ].add_action( a_desc, a_prob )
        
    def add_transition(self, s_hash, a_desc, snext_hash, t_prob=1.0, reward_obj=Reward( const=0.0 )):
        if s_hash not in self.define_statesD:
            self.define_statesD[ s_hash ] = DefineStateMoves( s_hash )
            
        self.define_statesD[ s_hash ].add_transition( a_desc, snext_hash, t_prob, reward_obj )
        
    def define_env_states_actions(self):
        """
        Will Set or Add prob and Reward entries to sn_probD and sn_rewardD
        
        action_prob controls the probability of picking an action from a list of actions.
        i.e. if in state s, there can be a list of (a1,p1), (a2,p2), (a3,p3), etc.
        
        trans_prob controls the probability of picking next state from a list of next states.
        i.e. if taking action a in state s, there can be a list of (sn1,p1), (sn2,p2), (sn3,p3), etc.
        
        The Reward object is always associated with (s,a,sn), however, it can vary with probabilty
        distributions of its own.
        """
        for (s_hash, DSM) in self.define_statesD.items():
            DSM.add_to_environment( self )
            
        # with TC updated, recalc terminal_set
        self.terminal_set, self.action_state_set = self.TC.get_terminal_set_and_action_set()
    
    def define_environment(self): # pragma: no cover
        """OVERRIDE THIS in order to define the environment."""
        
        # set up environment with calls to:
        # self.add_action( s_hash, a_desc, a_prob=1.0 )
        
        # self.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=Reward( const=0.0 ))
        #       ... OR ...         Reward Object can be replaced with constant float
        # self.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=0.0)
        
        # self.define_env_states_actions()
        
        #    -------------------
        # layout object is usually created here in child objects.
        #  
        #self.layout = GenericLayout( self, s_hash_rowL=None )
        
        self.start_state_hash = (0,0) # place holder

    def limited_start_state_list(self):
        """
        Return a limited list of starting states.
        Normally used by agents that need to discover the various
        states in an environment, like epsilon-greedy.
        
        OVERRIDE THIS to return a list of states smaller than 
        ALL ACTION STATES.
        """
        if self.defined_limited_start_state_list is None:
            return self.get_all_action_state_hashes()
        else:
            return self.defined_limited_start_state_list[:] # return a copy
    
    def define_limited_start_state_list(self, state_list ):
        # can define a start state list smaller than all action states
        # (call define_limited_start_state_list( state_list )
        self.defined_limited_start_state_list = state_list
    
    def get_start_state_hash(self):
        """Assume that the value of start_state_hash has been set... so now just return it."""
        if self.start_state_hash is None: # use a random state if none provided.
            return self.SC.get_random_state() # NOTE: this "fall-back" might return a terminal state.
        return self.start_state_hash

    def get_set_of_all_terminal_state_hashes(self):
        """
        Return a set of terminal state hash values. OR empty set.
        (No non-terminal states should be included.)
        Primarily used to detect the end of an episode.
        """
        # just to make sure it's "fresh", update terminal_set
        self.terminal_set, self.action_state_set = self.TC.get_terminal_set_and_action_set()
        return self.terminal_set

    def get_all_action_state_hashes(self):
        """
        Return a list of action state hash values. OR empty list.
        (No terminal states should be included.)
        """
        return [s_hash for s_hash in self.SC.iter_state_hash() if s_hash not in self.terminal_set]
        
    def get_any_action_state_hash(self):
        """
        Return a action state hash at random.
        """
        return random.choice( tuple( self.action_state_set ) )

    def get_action_snext_reward(self, s_hash, a_desc):
        """Get (next state hash, float reward) by taking action "a_desc" in state "s_hash" """
        sn_hash = self.TC.get_prob_weighted_next_state_hash( s_hash, a_desc )
        reward = self.TC.get_reward_value( s_hash, a_desc, sn_hash)
        
        return sn_hash, reward # (next state hash, float reward)

    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        (OR Empty list, if there are no actions)
        """
        return self.SAC.get_list_of_all_action_desc( s_hash, incl_zero_prob=True )

    def get_default_policy_desc_dict(self):
        """
        If the environment has a default policy, return it as a dictionary
            index=state_hash, value=action_desc
            
        NOTE: for deterministic policy, probability of each action is 1.0
              so do not need to return tuples of (action, probability)
        """
        # Policy Dictionary
        if self.default_policyD is None:
            return {}
        else:
            return self.default_policyD
    
    def get_num_states(self):
        return len(self.action_state_set) + len(self.terminal_set)
    
    def get_num_action_states(self):
        return len(self.action_state_set)
    
    def get_num_terminal_states(self):
        return len(self.terminal_set)
    
    
    def iter_all_action_states(self, randomize=False):
        """iterate over all action states in environment"""
        if randomize:
            for s_hash in random.sample( self.action_state_set, len(self.action_state_set) ):
                yield s_hash # assume none in terminal_set
        else:
            for s_hash in self.action_state_set:
                yield s_hash # assume none in terminal_set
    
    def iter_all_terminal_states(self):
        """iterate over all terminal states in environment"""
        for s_hash in self.terminal_set:
            yield s_hash # assume none in action_state_set
    
    def is_legal_state(self, s_hash):
        return s_hash in self.SC.stateD
    
    def is_terminal_state(self, s_hash):
        return s_hash in self.terminal_set
        
    def iter_all_states(self):
        """iterate over all states in environment"""
        for s_hash in self.iter_all_action_states():
            yield s_hash # assume none in terminal_set

        for s_hash in self.iter_all_terminal_states():
            yield s_hash # assume none in action_state_set

    def iter_state_hash_action_desc(self):
        """Iterate over all the (s,a) pairs in the environment"""
        for s_hash, a_desc in self.TC.transitionsD.keys():
            yield s_hash, a_desc

    def iter_action_desc_prob(self, s_hash, incl_zero_prob=False):
        """
        Iterate over all (action_desc, prob) pairs.
        if incl_zero_prob==True, include actions with zero probability.
        """
        for (a_desc, a_prob) in self.SAC.iter_action_desc_prob(s_hash, 
                                                 incl_zero_prob=incl_zero_prob):
            yield a_desc, a_prob

    def iter_next_state_prob_reward(self, s_hash, a_desc, incl_zero_prob=False):
        """
        Iterate over all (next_state_obj, prob, reward) tuples
        if incl_zero_prob==True, include actions with zero probability.
        """
        T = self.TC.get_transition_obj( s_hash, a_desc )
        for sn_hash, t_prob, reward in T.iter_sn_hash_prob_reward():
            yield sn_hash, t_prob, reward
    
    def get_layout_row_col_of_state(self, s_hash):# can be s_hash OR State object
        """
        --> OVERRIDE THIS FOR ANY SPECIALTY LAYOUTS <--
        
        Normally it's best to simply input "s_hash_rowL" to define layout.
        
        return an (i,j) tuple describing the location of s_hash in the layout.
        The upper left corner is (0,0) such that:
        i is the index to the row in "s_hash_rowL".
        """
        # in case a State object is input instead of s_hash, simply fix it
        if isinstance( s_hash, State ):
            s_hash = State.hash
                
        # some grid layouts can use this default (i,j)
        try:
            (i,j) = s_hash
            i = int(i)
            j = int(j)
            return i,j # (row, col)
        except:
            
            index = self.SC.get_state_co_index( s_hash )
            #print('for s_hash=',s_hash,'  index=',index)
            if index is None:
                return None, None # looks bad for (row, col) 
            
            n_states = len( self.SC )
            
            if n_states <= 16:
                return divmod( index, 4 )  # (row, col)
            else:
                len_row = 1 + int( math.sqrt( n_states ) )
                return divmod( index, len_row )  # (row, col)
            
        
    
    def get_estimated_rewards(self):
        """
        Return a dictionary of estimated rewards for each state.
        AND a dictionary of any special message
        (Will be exact for deterministic environment)
        """
        est_rD = {} # index=s_hash, value=float reward estimate.
        msgD   = {} # index=s_hash, value=any special message
        
        # initialize all rewards to zero for all states.
        for S in self.SC.iter_states():
            est_rD[ S.hash ] = RunningAve( S.hash )
        
        for s_hash, a_desc, T in self.TC.iter_all_transitions():
            for sn_hash, t_prob, reward in T.iter_sn_hash_prob_reward():
                Robj = T.get_reward_obj( sn_hash )

                if Robj.reward_type == CONST:
                    est_rD[ sn_hash ].add_val( reward  )
                    
                else:
                    msgD[ sn_hash ] = 'est'
                    # if the reward is stochastic, average 100 values
                    for i in range( 100 ):
                        est_rD[ sn_hash ].add_val( Robj()  )

                        
        # Need to convert RunningAve objects to float
        for (s_hash, RA) in est_rD.items():
            est_rD[s_hash] = RA.get_ave()
            #print(s_hash, RA)
            
        return est_rD, msgD

    def summ_print(self, long=True):
        print('___ "%s" Environment Summary ___'%self.name  )
        
        #term_set, action_set = self.TC.get_terminal_set_and_action_set()
        #print('Passing term_set ',term_set,' to StateColl summ_print.', type(term_set))
        self.SC.summ_print( terminal_set=self.terminal_set )
        
        self.AC.summ_print()
        if long:
            self.TC.summ_print()
        
        if self.layout is not None:
            self.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    
    def layout_print(self, vname='reward', fmt='', 
                     show_env_states=True, none_str='*'):
        """print the value "vname" formatted by the environment layout (if present). """
        
        if self.layout is None:
            print('...ERROR... "%s" tried to layout_print w/o a defined layout'%self.name )
            return
            
        if show_env_states:
            self.layout.s_hash_print( none_str=none_str )
        
        msgD = {} # initialize special message dictionary to empty
        
        if vname=='reward':
            valD, msgD = self.get_estimated_rewards() # index=s_hash, value=float reward estimate.
        else:
            valD = {} # empty if not recognized vname
            
        x_axis_label = self.layout.x_axis_label
        y_axis_label = self.layout.y_axis_label
        row_tickL =  self.layout.row_tickL
        col_tickL =  self.layout.col_tickL
            
        rows_outL = []
        for row in self.layout.s_hash_rowL:
            outL = []
            for s_hash in row:
                if s_hash not in self.SC.stateD:
                    if is_literal_str( s_hash ):
                        outL.append( s_hash[1:-1] )
                    else:
                        outL.append( none_str )
                else:
                    val = valD.get( s_hash, None )
                    if val is None:
                        outL.append( none_str )
                    else:
                        if fmt:
                            outL.append( fmt%val )
                        else:
                            outL.append( str(val) )
                if msgD.get( s_hash, ''):
                    outL[-1] = outL[-1] + msgD.get( s_hash, '')
                        
            rows_outL.append( outL )
        
        if rows_outL:
            print_string_rows( rows_outL,  const_col_w=True, 
                               line_chr='_', left_pad='    ', 
                               y_axis_label=y_axis_label, row_tickL=row_tickL, col_tickL=col_tickL,
                               header=self.name + ' %s Summary'%vname.title(), 
                               x_axis_label=x_axis_label, justify='right')
        
        
        


if __name__ == "__main__": # pragma: no cover
    
    E = EnvBaseline( mdp_file='Simple_Grid_World' )
    E.summ_print()
    
    
    E = EnvBaseline( mdp_file='Windy_Kings_Stoch_Gridworld' )
    E.summ_print()
    
