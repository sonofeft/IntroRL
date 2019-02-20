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
import pickle
import random

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties

    got_matplotlob = True
except:
    got_matplotlob = False


from introrl.state_actions import StateActions
from introrl.state_actions_coll import StateActionsColl
from introrl.utils.grid_funcs import print_string_rows, is_literal_str
from introrl.utils.gen_sort_key import NaturalOrStrKey
from introrl.utils.pylab_displays import draw_arrow

class Policy( StateActionsColl ):
    """
    Policy returns action or list of actions for each state in model/environment.
    
    A Policy object IS a StateActionsColl object (i.e. a collection of StateActions)
    It holds a StateActions object for each known State in the Environment.
    
    Each State object (unless a terminal state) has an action or collection of actions
    associated with it.
    
    All actions are taken with a probability in the range 0.0 to 1.0.
    If a state has only one action, it has a probability of 1.0
    If a state has >1 actions, the sum of their probabilities is 1.0    
        
    NOTE: A Policy has its own "state_coll" and "action_coll" (State and Action Collections)
    Ideally, they are the same as the Environment "state_coll" and "action_coll" but 
    These might differ... They are the Agents view of the world.
    """
    
    def __init__(self, name='Policy', environment=None):
        
        #if environment is None:
        #    raise ValueError( 'No "environment" specified for Policy' )
        
        """ 
        Policy for deciding actions by an Agent in an Environment. 

          INHERITED COMMON METHODS from StateActionsColl
          
        def get_prob_weighted_action(self, state_hash):
        def set_action_prob(self, state_hash, action_desc, prob=1.0):
        def set_sole_action(self, state_hash, action_desc):
        def initialize_sole_random(self, state_hash):
        def intialize_to_equiprobable(self, state_hash):
        def has_action(self, state_hash, action_desc):
        def get_action_prob(self, state_hash, action_desc):
        def remove_action(self, state_hash, action_desc):
        def get_list_of_all_action_desc_prob(self, state_hash, incl_zero_prob=False):
        def get_list_of_all_action_desc(self, state_hash, incl_zero_prob=False):
        def iter_action_desc_prob(self, state_hash, incl_zero_prob=False):
        def get_state_action_obj(self, state_hash):
        def add_state_action(self, state_hash):
        """
        
        StateActionsColl.__init__(self, name=name  )
        
        if environment is not None:
            """
            Teach policy about all states and actions directly from the Environment object.
            Unless otherwise modified, policy will be an equiprobable policy over all
            actions from each state.
            """
            self.learn_all_states_and_actions_from_env( environment )
    
    def make_dict_of_policy(self):
        """
        Return a dictionary describing the policy
        """
        policyD = {} # index=s_hash, value=a_desc
        # Iterate over all states. Return state_hash
        for s_hash in self.iter_all_policy_states():
            a_desc =  self.get_single_action( s_hash )
            if a_desc is not None:
                policyD[ s_hash ] = a_desc
        return policyD
    
    def make_pickle_filename(self, fname):
        """Make a file name ending with .bb_pickle """
        if fname is None:
            fname = self.name.replace(' ','_') + '.pi_pickle'
            
        else:
            fname = fname.replace(' ','_').replace('.','_') + '.pi_pickle'
            
        return fname
    
    def save_to_pickle_file(self, fname=None): # pragma: no cover
        """Saves data to pickle file."""
        # make a policy dictionary
        policyD = self.make_dict_of_policy()
        
        # build name for pickle
        fname = self.make_pickle_filename( fname )
        
        saveD = {}
        saveD['name'] = self.name
        saveD['policyD'] = policyD
        
        fileObject = open(fname,'wb')
        pickle.dump(saveD,fileObject, protocol=2)# protocol=2 is python 2&3 compatible.
        fileObject.close()
        print('Saved Policy to file:',fname)
    
    def read_pickle_file(self, fname=None): # pragma: no cover
        """Reads data from pickle file."""
        
        fname = self.make_pickle_filename( fname )
        if not os.path.isfile( fname ):
            print('Pickle File NOT found:', fname)
            return False
        
        fileObject = open(fname,'rb')  
        readD = pickle.load(fileObject)  
        
        self.name = readD['name']
        policyD = readD['policyD']
        
        fileObject.close()
        print('Read Policy from file:',fname)
        
        return policyD

    def init_from_pickle_file(self, fname=None): # pragma: no cover
        """Initialize policy from policy pickle file."""
        policyD = self.read_pickle_file( fname=fname )
        if policyD:
            self.set_policy_from_piD( policyD )
        
    def set_policy_at_state_hash(self, state_hash, policy_a_desc ):
        """Set the policy at state_hash, to be action, policy_a_desc."""
        
        if self.has_action( state_hash, policy_a_desc):
            self.set_sole_action( state_hash, policy_a_desc)
        else:
            print('ERROR... tried to set policy for state=%s to action=%s'%(str(state_hash), str(policy_a_desc)) )
            print('   That (state, action) pair is not defined in the Environment')
            raise ValueError( '(state, action) pair not recognized.' + '(%s,%s)'%(str(state_hash), str(policy_a_desc)) )
            

    def set_policy_from_piD(self, init_policyD): # set policy with state_hash, action_desc
        """        
        init_policyD: will be used to initialize policy action/prob pairs
           ASSUME: init_policyD: index=state_hash: 
                                 value=action_desc
        
        """
        # initialize with input policy
        # NOTE: init_policyD can take 3 different forms
        #       init_policyD: index=state_hash: 
        #                     value=action_desc 
        
        #print('init_policyD[(25, 7, 0, 1)] =',init_policyD[(25, 7, 0, 1)] )
        #print('----------------- Starting Policy Initialize -------------------------')
        
        # start out by setting everything to 0.0
        self.intialize_to_all_zero_prob() # <--------------------- NOTE: setting all to zero.
        
        for (s_hash, ainp) in init_policyD.items():
            # ainp is simply an a_desc
            #self.add_state_action( s_hash )
            #self.set_action_prob( s_hash, ainp, prob=1.0)
            
            self.set_sole_action( s_hash, ainp)
            
            #self.get_state_action_obj( s_hash ).summ_print()
                
        #print('----------------- Finished Policy Initialize -------------------------')
        #s_hash = (25, 7, 0, 1)
        #print( 'self.get_single_action( %s ) ='%str(s_hash),self.get_single_action( s_hash ) )
        #print( 'init_policyD[%s] ='%str(s_hash),init_policyD[s_hash] )
        #print( self.get_state_action_obj( s_hash ) )
        #self.get_state_action_obj( s_hash ).summ_print()

    def learn_all_states_and_actions_from_env(self, env):
        """
        Teach policy about all states and actions directly from the Environment object.
        Unless otherwise modified, policy will be an equiprobable policy over all
        actions from each state.
        """
        #self.intialize_policy_to_equiprobable( env )
        #self.learn_a_legal_action_from_env( env )
        
        for s_hash in env.iter_all_action_states():
            self.state_coll.add_state( s_hash )
            
            aL = env.get_state_legal_action_list( s_hash )
            for a_desc in aL:
                self.action_coll.add_action( a_desc )
                self.set_action_prob( s_hash, a_desc, prob=1.0)
        
        # learn all states... fill new state_coll with new State objects
        #for S in env.SC.iter_states():
        #    self.state_coll.add_state( S.hash )
            
        # learn all actions... fill new action_coll with new Action objects.
        #for A in env.AC.iter_actions():
        #    self.action_coll.add_action( A.desc )

    def learn_a_legal_action_from_env(self, env):
        """
        Learn a legal action picked randomly from each state's possible actions.
        The legal action list is taken from the environment.
        """
        for s_hash in env.iter_all_action_states():
            aL = env.get_state_legal_action_list( s_hash )
            if aL:
                a_desc = random.choice( aL )
                self.set_sole_action( s_hash, a_desc)
                
    def intialize_policy_to_random(self, env=None):
        """
        Initialize entire policy to random sole action
        If input, use env to define possible actions.
        """
        if env is not None:
            self.learn_a_legal_action_from_env( env )
        
        for SA in self.state_actionsD.values():
            SA.initialize_sole_random()
            
    def intialize_policy_to_equiprobable(self, env=None):
        """
        Initialize entire policy to all (s,a) having equal probability
        If input, use env to define possible actions.
        """
        if env is not None:
            self.learn_a_legal_action_from_env( env )
        
        for SA in self.state_actionsD.values():
            SA.intialize_to_equiprobable()
    
    def iter_policy_ap_for_state(self, state_hash, incl_zero_prob=False):
        """Iterate over all actions of state. return (action_desc, probability)"""
        for (a_desc, p) in self.iter_action_desc_prob( state_hash, incl_zero_prob=incl_zero_prob):
            yield a_desc, p
    
    def iter_all_policy_states(self):
        """Iterate over all states. Return state_hash"""
        for S in self.state_coll.iter_states():
            yield S.hash
                    
    def get_single_action(self,  state_hash ):
        """
        Return an Action description for the given state_hash.
        """
        #print('self.state_coll.has_state_hash( "%s" )'%str(state_hash), self.state_coll.has_state_hash( state_hash ))
        A = self.get_prob_weighted_action( state_hash )
        if A is not None:
            return A.desc
        else:
            return A
    
    def save_diagram(self, environment, inp_colorD=None, pad=0.1, save_name='', 
                     show_arrows=True, do_show=False, scale=1.0, h_over_w=1.0,
                     show_terminal_labels=True):
        """
        Use matplotlib to create a color-coded policy diagram.
        Requires an environment.layout to do it.
        
        if inp_colorD is provided, it has, index=action, value=color string.
        """
        if (environment is None) or (environment.layout is None):
            print('WARNING... Need an environment with a layout to create a policy diagram.')
            return
        
        if not got_matplotlob:
            print('WARNING... Need matplotlib to create a policy diagram... it failed to import.')
            return
        
        #colorL = ['r','g','b','m','c','y']
        colorL = ['r','g','b','m','c','y',
                  'darkcyan','deepskyblue','darkorange','brown','deeppink',
                  'maroon','crimson','seagreen','fuchsia','darkviolet' ]

        colorD = {} # index=action, value=color string
        if inp_colorD is not None:
            colorD.update( inp_colorD )
        
        Ncols = len( environment.layout.s_hash_rowL[0] )
        Nrows = len( environment.layout.s_hash_rowL )
        
        w_lr = 1.0
        h_tb = 1.0
        fig = plt.figure( figsize=( scale*(Ncols+w_lr), h_over_w*scale*(Nrows+h_tb)) )
        
        axs = fig.add_axes()
        plt.axes()
        
        alignment = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
        font = FontProperties()
        font.set_size('large')
        font.set_family('fantasy')
        font.set_style('normal')
        
        d = 1.0 - pad
        d2 = d / 2.0
        
        arrowL = [] # list of (s_hash, a_desc, color) for all arrows
        
        def get_rect_color( s_hash, action_list ):
            if len(action_list)==0:
                return ''
                
            if len(action_list)==1:
                a_desc = action_list[0]
                s = str( a_desc )
                if s in colorD:
                    c = colorD[s]
                else:
                    c = colorL[ len(colorD) % len(colorL) ]
                    colorD[s] = c
                return c
            
            basic_color = 'skyblue'
            if environment.layout.colorD is not None:
                if environment.layout.basic_color:
                    basic_color = environment.layout.basic_color
                return environment.layout.colorD.get(s_hash, basic_color)
            return basic_color
                
        
        for irow,row in enumerate(environment.layout.s_hash_rowL):
            outL = []
            y = Nrows - irow - 1
            for jcol,s_hash in enumerate(row):
                
                if environment.is_legal_state( s_hash ):
                    #a_desc = self.get_single_action(s_hash)
                    actionL = [a for (a,p) in self.get_list_of_all_action_desc_prob( s_hash, incl_zero_prob=False)]
                    
                    c_rect = get_rect_color( s_hash, actionL )
                    
                    if not actionL:
                        # if no actions possible, simply put state label.
                        # (if a color is specified, put a colored rectangle as well)
                        try:
                            c_rect = environment.layout.colorD.get(s_hash, '')
                        except:
                            c_rect = ''
                        if c_rect:
                            rect = Rectangle((jcol,y), d,   d, fc=c_rect, alpha=0.5, edgecolor=c_rect)
                            plt.gca().add_patch( rect )
                            
                        if show_terminal_labels:
                            s = str( s_hash )
                            t = plt.text(jcol+d2, y+d2, s, fontproperties=font, **alignment)
                    else:
                        # get here if the policy has one or more actions in this state.
                        
                        #      Rectangle(  (x,y),    width,   height)
                        rect = Rectangle((jcol,y), d,   d, fc=c_rect, alpha=0.5, edgecolor=c_rect)
                        plt.gca().add_patch( rect )
                        
                        sL = []
                        for a_desc in actionL:
                            s = str( a_desc )
                            sL.append(s)
                            if s in colorD:
                                c = colorD[s]
                            else:
                                c = colorL[ len(colorD) % len(colorL) ]
                                colorD[s] = c
                            #print('a_desc=',a_desc,' color=',c)
                            
                            # build a list of arrows to be placed on top after all rectangles are made.
                            if show_arrows:
                                arrowL.append( (s_hash, a_desc, c) )
                        
                        if sL:
                            t = plt.text(jcol+d2, y+d2, ' '.join(sL), fontproperties=font, **alignment)
                             
                            
                else:
                    rect = Rectangle((jcol,y), d,   d, fc='gray', alpha=0.5, edgecolor='gray')
                    plt.gca().add_patch( rect )
                    
                    if is_literal_str( s_hash ):
                        t = plt.text(jcol+d2, y+d2, s_hash[1:-1], fontproperties=font, **alignment)
    
    
        plt.xlim(0, Ncols)
        plt.ylim(0, Nrows)
        # if any arrows being shown, do it.
        if arrowL:
            for (s_hash, a_desc, c) in arrowL:
                try:
                    # draw_arrow relies on environment.get_action_snext_reward to calc next state.
                    #  (if it is stochastic, that will result in a random arrow)
                    draw_arrow(plt, Nrows, environment, s_hash, a_desc, pad,
                               color=c, Rinner=0.25, frac_len=0.2)
                except:
                    print('draw_arrow FAILED for:',s_hash, a_desc)
                
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        
        plt.title('Policy for ' + environment.name )
        
        try:
            plt.tight_layout()
        except:
            print('WARNING... plt.tight_layout() FAILED.')

        if save_name:
            if save_name.lower().endswith('.png'):
                fig.savefig( save_name )
            else:
                fig.savefig( save_name + '.png' )
                
        if do_show:
            plt.show()
        
    
    def summ_print(self, verbosity=2, environment=None, 
                   show_env_states=True, none_str='*'): # pragma: no cover
        """Show State objects in sorted state_hash order."""
        print('___ Policy Summary ___' )
        print('    Nstate-actions=%i'%len(self.state_actionsD) )
        
        #self.state_coll.summ_print()
        #self.action_coll.summ_print()
        sL = sorted( [(S.hash,S) for S in self.state_actionsD.keys()], key=NaturalOrStrKey )
        if verbosity==2:
            for s_hash,S in sL:
                SA = self.state_actionsD[ S ]
                SA.summ_print()
                exL = [str(self.get_single_action(S.hash)) for i in range(16) ]
                print('        ex. actions:', ' '.join(exL))
        elif verbosity==1:
            print('        State Action')
            for s_hash,S in sL:
                SA = self.state_actionsD[ S ]
                
                # force a single action
                a_desc = self.get_single_action(S.hash)
                
                print('%13s'%str(s_hash),' %s'%a_desc, end=' ')
                if len(SA)>1:
                    optL = sorted( [ A.desc for (A,prob) in SA.action_probD.items()], key=NaturalOrStrKey )
                    print('from:',', '.join(optL))
                else:
                    print()
            
        
        if (environment is not None) and  (environment.layout is not None):
            # make summ_print using environment.layout
            if show_env_states:
                environment.layout.s_hash_print( none_str='*' )
            
            
            rows_outL = []
            for row in environment.layout.s_hash_rowL:
                outL = []
                for s_hash in row:
                    if not environment.is_legal_state( s_hash ):
                        if is_literal_str( s_hash ):
                            outL.append( s_hash[1:-1] )
                        else:
                            outL.append( none_str )
                    else:
                        a_desc = self.get_single_action(s_hash)
                        if a_desc is None:
                            outL.append( '  *' )
                        else:
                            outL.append( self.get_state_summ_str( s_hash, verbosity=verbosity ) )
                            
                rows_outL.append( outL )
            
            
            row_tickL = environment.layout.row_tickL
            col_tickL = environment.layout.col_tickL
            y_axis_label = environment.layout.y_axis_label
            
            if not environment.layout.x_axis_label:
                x_axis_label = 'Actions'
            else:
                x_axis_label = environment.layout.x_axis_label
            
            print_string_rows( rows_outL, row_tickL=row_tickL, const_col_w=True, 
                               line_chr='_', left_pad='    ', 
                               header=environment.name + ' Policy Summary', 
                               x_axis_label=x_axis_label, justify='right',
                               col_tickL=col_tickL, y_axis_label=y_axis_label)
            

if __name__ == "__main__": # pragma: no cover
    
    from introrl.mdp_data.sample_gridworld import get_gridworld
    
    gridworld = get_gridworld()
    policyD = gridworld.get_default_policy_desc_dict()
    
    pi = Policy( environment=gridworld )
    #pi.learn_all_states_and_actions_from_env( gridworld )
    #pi.set_policy_from_piD( policyD )
    #pi.intialize_policy_to_random( env=gridworld )
    #pi.intialize_policy_to_equiprobable( env=gridworld )
    #pi.set_policy_at_state_hash( (0,0), 'D' )
    #pi.set_policy_from_piD( policyD )

    #       init_policyD: index=state_hash: 
    #                     value=action_desc OR 
    #                     value=list of action_desc OR
    #                     value=list of (action_desc, prob) OR [action_desc, prob]

    #piD = {(0, 0):('R','L') }
    #piD = {(0, 0):[('R',0.6), ('D',0.4)] }
    #pi.set_policy_from_piD( piD )
    
    #pi.summ_print( environment=gridworld )
    
    pi.save_diagram( gridworld, #inp_colorD={'U':'crimson', 'D':'pink', 'L':'plum', 'R':'orange'}, 
                     save_name='sample_gridworld_policy', do_show=True, scale=0.75, h_over_w=1.0)
    