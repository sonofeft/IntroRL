
from introrl.layouts.generic_layout import GenericLayout
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

def get_random_walk():

    env = EnvBaseline( name='Random Walk MRP' ) # GenericLayout set below
    env.set_info( 'Random Walk MRP' )

    actionD = {'A':  ('L','R'),
               'B':  ('L','R'),
               'C':  ('L','R'),
               'D':  ('L','R'),
               'E':  ('L','R') }
                   
    rewardD = {'Win': 1.0, 'Lose':0.0}

    for (s_hash, moveL) in actionD.items():
        for a_desc in moveL:
            env.add_action( s_hash, a_desc, a_prob=1.0 )

    def add_event( s_hash, a_desc, sn_hash ):
        #print('s_hash, a_desc, sn_hash',s_hash, a_desc, sn_hash)
        r = rewardD.get( sn_hash, 0.0)
        env.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=r)

    mrpL = ['Lose','A','B','C','D','E','Win']
    for i,ci in enumerate( mrpL[1:-1] ):
        add_event( ci, 'L', mrpL[i] )
        add_event( ci, 'R', mrpL[i+2] )

    env.define_env_states_actions()  # send all states and actions to environment

    # --------------------
        
    s_hash_rowL=[ mrpL ]
        
    env.layout = GenericLayout( env, s_hash_rowL=s_hash_rowL )

    env.start_state_hash = 'C'

    # define default_policyD
    policyD = {} # index=state_hash, value=action_desc

    policyD[ 'A' ] = ('L','R')
    policyD[ 'B' ] = ('L','R')
    policyD[ 'C' ] = ('L','R')
    policyD[ 'D' ] = ('L','R')
    policyD[ 'E' ] = ('L','R')

    env.default_policyD = policyD
    
    return env

if __name__ == "__main__": # pragma: no cover
    
    env = get_random_walk()
    env.summ_print()
    #env.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    env.save_to_pickle_file( fname=None )
