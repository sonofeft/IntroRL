
from introrl.layouts.generic_layout import GenericLayout
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

#  example 7.1 uses a 19 state random walk MRP
def get_random_walk( Nside_states=9, win_reward=1.0, lose_reward=-1.0, step_reward=0.0 ): 

    Nstates = 2 * Nside_states + 1

    s = '(L%i, R%i)'%(Nside_states, Nside_states)
    env = EnvBaseline( name='%i State Random Walk MRP'%Nstates + s ) # GenericLayout set below
    env.set_info( '%i State Random Walk MRP'%Nstates + s )
    
    RstateL = ['R+%i'%i for i in range(1, Nside_states+1)]
    LstateL = list(reversed([s.replace('R+','L-') for s in RstateL]))
    
    actionD = {}
    for s in LstateL:
        actionD[s] = ('L','R')
    actionD['C'] = ('L','R')
    for s in RstateL:
        actionD[s] = ('L','R')
    
    rewardD = {'Win': win_reward, 'Lose':lose_reward}

    for (s_hash, moveL) in actionD.items():
        for a_desc in moveL:
            env.add_action( s_hash, a_desc, a_prob=1.0 )

    def add_event( s_hash, a_desc, sn_hash ):
        #print('s_hash, a_desc, sn_hash',s_hash, a_desc, sn_hash)
        r = rewardD.get( sn_hash, step_reward)
        env.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=r)

    mrpL = ['Lose']+LstateL+['C']+RstateL+['Win']
    for i,ci in enumerate( mrpL[1:-1] ):
        add_event( ci, 'L', mrpL[i] )
        add_event( ci, 'R', mrpL[i+2] )

    env.define_env_states_actions()  # send all states and actions to environment

    # -------------------- make layout for printing ------------------
        
    s_hash_rowL=[ mrpL ]
        
    env.layout = GenericLayout( env, s_hash_rowL=s_hash_rowL )

    env.start_state_hash = 'C'

    # define default_policyD
    policyD = {} # index=state_hash, value=action_desc

    policyD[ 'C' ] = ('L','R')
    for s in LstateL:
        policyD[s] = ('L','R')
    for s in RstateL:
        policyD[s] = ('L','R')

    env.default_policyD = policyD
    
    return env

if __name__ == "__main__": # pragma: no cover
    
    env = get_random_walk()
    env.summ_print()
    #env.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    env.save_to_pickle_file( fname=None )
