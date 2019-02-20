
from introrl.layouts.generic_layout import GenericLayout
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward


def get_env():

    env = EnvBaseline( name='Simple Six State World' )
    env.set_info( 'Simple Six State World' )

    actionD = {'A':  ('U',),
               'B':  ('ur', 'D'),
               '<C>': ('ur', 'dl'),
               'D':  ('ur', 'ul') }
                   
    rewardD = {'A': -1.0, 'E': 0.5, 'F':1.0}


    for (s_hash, moveL) in actionD.items():
        for a_desc in moveL:
            env.add_action( s_hash, a_desc, a_prob=1.0 )

    def add_event( s_hash, a_desc, sn_hash ):
        r = rewardD.get( sn_hash, 0.0)
        env.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=r)

    add_event( 'A', 'U', 'B' )
    #add_event( 'A', 'Te', 'E' )
    add_event( 'B', 'D', 'A' )
    add_event( 'B', 'ur', '<C>' )
    add_event( '<C>', 'dl', 'B' )
    add_event( '<C>', 'ur', 'D' )
    add_event( 'D', 'ur', 'F' )
    add_event( 'D', 'ul', 'E' )

    env.define_env_states_actions()  # send all states and actions to environment

    # --------------------

    s_hash_rowL =[('*',   'E',   '*',   'F'),
                  ('*',   '*',   'D',   '*'),
                  ('*', '<C>',   '*',   '*'),
                  ('B',   '*',   '*',   '*'),
                  ('A',   '*',   '*',   '*')]
    
    env.layout = GenericLayout( env , s_hash_rowL=s_hash_rowL )



    env.start_state_hash = '<C>'

    # define default_policyD
    policyD = {} # index=state_hash, value=action_desc

    policyD[ 'B' ] = 1
    policyD[ '<C>' ] = 1
    policyD[ 'D' ] = 1

    env.default_policyD = policyD
    
    return env

if __name__ == "__main__": # pragma: no cover
    
    env = get_env()
    env.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
