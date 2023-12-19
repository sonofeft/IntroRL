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
from itertools import product

from introrl.utils.tiles_infinite import Tile
from introrl.black_box_sims.sim_baseline import Simulation

class TabularSimulation( Simulation ):
    """
    Wrap a ContinuousSimulation object to make a tabular model from it.
    """
    
    def __init__(self, cont_sim, num_regionsL=None, edge_fracL=None):
        
        self.cont_sim = cont_sim
        self.name = cont_sim.name
        self.n_state_vars = len( cont_sim.paramL )
        
        self.info = """A Tabular wrapper for the ContinuousSimulation: %s."""%cont_sim.name
        
        # build description of state space        
        if num_regionsL is None:
            num_regionsL = [5] * self.n_state_vars
        while len(num_regionsL) < self.n_state_vars:
            num_regionsL.append( num_regionsL[-1] )
        self.num_regionsL = num_regionsL
        
        # look at edge_fracL to determine size of edge states
        if edge_fracL is None:
            edge_fracL = [0.5] * self.n_state_vars
        while len(edge_fracL) < self.n_state_vars:
            edge_fracL.append( edge_fracL[-1] )
        
        # set range of Tile Dimension objects
        self.lo_valL = []
        self.hi_valL = []
        
        self.true_lo_valL = [] # actual limit of parameter in cont_sim
        self.true_hi_valL = []
        
        for i,p in enumerate( cont_sim.paramL ):
            delta = (p.max_value - p.min_value) / float(self.num_regionsL[i])
            edge_offset = delta * edge_fracL[i]
            
            self.lo_valL.append( p.min_value + edge_offset )
            self.hi_valL.append( p.max_value - edge_offset )
            
            self.true_lo_valL.append( p.min_value )
            self.true_hi_valL.append( p.max_value )
            
        
        # build a tile object from which to make tables of states.
        self.tile = Tile( lo_valL=self.lo_valL, hi_valL=self.hi_valL, 
                          num_regionsL=self.num_regionsL)
                
        # build the generic layout
        tupL = sorted( self.tile.rev_state_indexD.keys(), reverse=True )
        last_key_0 = -1
        s_hash_rowL = []
        row_tickL = []
        for tup in tupL:
            if tup[0] != last_key_0:
                s_hash_rowL.append( [] )
                last_key_0 = tup[0]
                row_tickL.append( '%g:%g'%self.tile.dimL[0].get_region_range( tup[0] ) )
                
            s_hash_rowL[-1].append( tup )
        
        y_axis_label = cont_sim.paramL[0].name
        if len(cont_sim.paramL) > 1:
            x_axis_label = ', '.join( [p.name for p in cont_sim.paramL[1:]]  )
                
            col_tickL = [ '%g:%g'%self.tile.dimL[1].get_region_range(i) for i in range(self.tile.dimL[1].num_regions)]
        else:
            x_axis_label = ''
            col_tickL = None
        
        
        Simulation.__init__(self, name=self.name, s_hash_rowL=s_hash_rowL, 
                            row_tickL=row_tickL, col_tickL=col_tickL, 
                            x_axis_label=x_axis_label, y_axis_label=y_axis_label, 
                            colorD=None, basic_color='',
                            start_time=0)
        
        # figure out terminal_set
        self.terminal_set = set()
        for tup in tupL:
            
            s_nominal = [d.get_nominal_value(irange) for d,irange in zip(self.tile.dimL, tup)]
            if cont_sim.is_terminal_state( s_vector=s_nominal ):
                #print('Add terminal_set:', s_nominal, tup)
                self.terminal_set.add( tup )
                continue
            
            
            rangeL = [d.get_region_range(irange) for d,irange in zip(self.tile.dimL, tup)]
            for s_vector in product( *rangeL ):
                if cont_sim.is_terminal_state( s_vector=s_vector ):
                    #print('...Add terminal_set:', s_vector, tup, rangeL)
                    self.terminal_set.add( tup )
                    continue
        
        # state hash
        self.action_state_set = set( tupL ) - self.terminal_set # a set of state hashes
        
        #print('self.terminal_set:',self.terminal_set)
        #print()
        #print('self.action_state_set:',self.action_state_set)
        
        # estimate start state
        s_vector = cont_sim.get_s_vector()
        s_tup = tuple( self.tile.get_regions(s_vector) )
        self.start_state_hash = s_tup
        
        # for generating sample state, action data, keep track of get_next_s_vector calls
        self.next_s_vectorD = {} # index=s_hash, value=s_vector (randomly generated)
        
    def get_next_s_vector(self, s_hash, make_new_vector=False):
        
        if (s_hash in self.next_s_vectorD) and not make_new_vector:
            return self.next_s_vectorD[ s_hash ]
        
        rangeL = [d.get_region_range(irange) for d,irange in zip(self.tile.dimL, s_hash)]
        minL = [r[0] for r in rangeL]
        maxL = [r[1] for r in rangeL]
        
        # limit range to true range of cont_sim parameters
        minL = [max(self.true_lo_valL[i], minL[i]) for i in range(len(minL))]
        maxL = [min(self.true_hi_valL[i], maxL[i]) for i in range(len(maxL))]
        
        deltaL = [ (maxL[i]-minL[i]) for i in range(len(minL))]
        edge_offsetL = [d*0.0005 for d in deltaL]
        
        def one_or_zero():
            if random.random()>0.5:
                return 1.0
            else:
                return 0.0
            
        s_vector = [mn + ed + random.random()*d for mn,ed,d in zip(minL, edge_offsetL, deltaL) ]
        self.next_s_vectorD[ s_hash ] = s_vector
        return s_vector
    
    def get_nominal_s_vector(self, s_hash):
        return [d.get_nominal_value(irange) for d,irange in zip(self.tile.dimL, s_hash)]
                            
    def get_action_snext_reward(self, s_hash, a_desc):
        """
        Return  snext_hash, reward
        """
        #s_nominal = self.get_nominal_s_vector( s_hash )
        s_nominal = self.get_next_s_vector( s_hash, make_new_vector=True )
        
        sn_vector, reward = self.cont_sim.get_action_snext_reward( a_desc, s_vector=s_nominal)
        
        sn_hash = tuple( self.tile.get_regions( sn_vector ) )
        #def f(L):
        #    return ','.join( ['%7.3f'%v for v in L] )
        #print( 'for s_hash=',s_hash,'  s_nominal=',f(s_nominal), '  sn_vector=',f(sn_vector), '   sn_hash=',sn_hash )
        
        return sn_hash, reward
        
    def get_state_legal_action_list(self, s_hash):
        """
        Return a list of possible actions from this state.
        Include any actions thought to be zero probability.
        OR Empty list, if the agent must simply guess.
        """
        if s_hash in self.action_state_set:
            #s_nominal = self.get_nominal_s_vector( s_hash )
            s_nominal = self.get_next_s_vector( s_hash, make_new_vector=True )
            return self.cont_sim.get_state_legal_action_list( s_vector=s_nominal )
        else:
            return []

        

if __name__=="__main__":
    
    import time
    from introrl.continuous_sims.sim_continuous import ContinuousSimulation
    from introrl.agent_supt.model import Model
    from introrl.environments.env_baseline import EnvBaseline
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    
    start_time = time.time()
    
    MCar = ContinuousSimulation( name='Mountain Car', step_reward=-1.0)
    
    TSim = TabularSimulation( MCar, num_regionsL=[120,120], edge_fracL=[.01, 0.5] )
    #TSim.layout.s_hash_print( none_str='*' )
    
    model = Model( TSim, build_initial_model=True )
    
    model.collect_transition_data( num_det_calls=100, num_stoic_calls=100 )
    #model.summ_print( long=True )
    
    print('_'*55)
    env = EnvBaseline( s_hash_rowL=TSim.s_hash_rowL, 
                       row_tickL=TSim.row_tickL, col_tickL=TSim.col_tickL,
                       x_axis_label=TSim.x_axis_label, y_axis_label=TSim.y_axis_label )
    model.add_all_data_to_an_environment( env )
    
    print('_'*55)    
    #env.summ_print( long=True )
    
    policy, state_value = dp_value_iteration( env, do_summ_print=True, fmt_V='%.2f', fmt_R='%.2f',
                                              max_iter=1000, err_delta=0.000001, 
                                              gamma=0.99, iteration_prints=10)

    diag_colorD = { '1':'g', '0':'r', '-1':'b'}

    policy.save_diagram( env, inp_colorD=diag_colorD, pad=0.1, save_name='mountain_car_policy', 
                     show_arrows=False, do_show=True, scale=1.0, h_over_w=1.0,
                     show_terminal_labels=True)
    
    print( 'Total Time =',time.time() - start_time )
    
    env.save_to_pickle_file('mountain_car_env')
    
    
    