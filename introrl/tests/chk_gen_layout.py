#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

from introrl.environments.env_baseline import EnvBaseline
from introrl.layouts.generic_layout import GenericLayout

class SixStateEnvironment( EnvBaseline ):
    
    def __init__(self, name='Layout Check' ):
        
        EnvBaseline.__init__(self, name=name)
        
    def define_environment(self):
        
        # possible moves: ('U','ur','R','dr','D','dl','L','ul','Te')
        actionD = {'A':  ('U',),#'Te'),
                   'B':  ('ur', 'D'),
                   '<C>': ('ur', 'dl'),
                   'D':  ('ur', 'ul') }
                       
        rewardD = {'A': -1.0, 'E': 0.5, 'F':1.0}
        
        
        for (s_hash, moveL) in actionD.items():
            for a_desc in moveL:
                self.add_action( s_hash, a_desc, a_prob=1.0 )
        
        def add_event( s_hash, a_desc, sn_hash ):
            r = rewardD.get( sn_hash, 0.0)
            self.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=r)
        
        add_event( 'A', 'U', 'B' )
        #add_event( 'A', 'Te', 'E' )
        add_event( 'B', 'D', 'A' )
        add_event( 'B', 'ur', '<C>' )
        add_event( '<C>', 'dl', 'B' )
        add_event( '<C>', 'ur', 'D' )
        add_event( 'D', 'ur', 'F' )
        add_event( 'D', 'ul', 'E' )
        
        self.define_env_states_actions()  # send all states and actions to environment

        s_hash_rowL =[('*',   'E',   '*',   'F'),
                      ('*',   '*',   'D',   '*'),
                      ('*', '<C>',   '*',   '*'),
                      ('B',   '*',   '*',   '*'),
                      ('A',   '*',   '*',   '*')]
        
        self.layout = GenericLayout( self , s_hash_rowL=s_hash_rowL )

        
if __name__ == "__main__": # pragma: no cover
    
    env = SixStateEnvironment()
    env.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    
