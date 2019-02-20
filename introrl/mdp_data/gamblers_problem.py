
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward


# --------------------
# define layout for output
    
s_hash_rowL = [] # layout rows for makeing 2D output

s_hash_rowL.append( [0, '*', '*', '*', '*', '*', '*', '*', '*', '*'] ) # use append to put origin at upper left
s_hash_rowL.append( [1,  2,  3,  4,  5,  6,  7,  8,  9, 10 ] )
s_hash_rowL.append( [11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ] )
s_hash_rowL.append( [21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ] )
s_hash_rowL.append( [31, 32, 33, 34, 35, 36, 37, 38, 39, 40 ] )
s_hash_rowL.append( [41, 42, 43, 44, 45, 46, 47, 48, 49, 50 ] )
s_hash_rowL.append( [51, 52, 53, 54, 55, 56, 57, 58, 59, 60 ] )
s_hash_rowL.append( [61, 62, 63, 64, 65, 66, 67, 68, 69, 70 ] )
s_hash_rowL.append( [71, 72, 73, 74, 75, 76, 77, 78, 79, 80 ] )
s_hash_rowL.append( [81, 82, 83, 84, 85, 86, 87, 88, 89, 90 ] )
s_hash_rowL.append( [91, 92, 93, 94, 95, 96, 97, 98, 99, 100] )      
    
#gambler.layout = GenericLayout( gambler, s_hash_rowL=s_hash_rowL )



def get_gambler(prob_heads=0.4):

    gambler = EnvBaseline( name='Gamblers Coin Flip Problem',
                           s_hash_rowL=s_hash_rowL,
                           colorD={100:'g', 0:'r'},
                           basic_color='skyblue' )
    gambler.set_info( 'Example 4.3 from Sutton & Barto 2nd Edition page 84.' )

    for s in range(1, 100): # 1 to 99
        s_max = min(s, 100-s)
        for a_desc in range(1, s_max + 1):
            gambler.add_action( s, a_desc, a_prob=1.0 )

    # define reward for all states
    def get_reward( sn ):
        if sn==100:
            return 1.0
        else:
            return 0.0

    # define all possible transitions.
    for s in range(1, 100): # 1 to 99
        s_max = min(s, 100-s)
        for a_desc in range(1, s_max + 1):
            sn_hash = s - a_desc
            rval = get_reward( sn_hash )
            gambler.add_transition( s, a_desc, sn_hash, t_prob=1.0-prob_heads, reward_obj=rval)

            sn_hash = s + a_desc
            rval = get_reward( sn_hash )
            gambler.add_transition( s, a_desc, sn_hash, t_prob=prob_heads, reward_obj=rval)
            
    gambler.define_env_states_actions()  # send all states and actions to environment

    # If there is a start state, define it here.
    gambler.start_state_hash = (50)

    # define default policy (if any)
    gambler.default_policyD = {}
    
    return gambler

if __name__ == "__main__": # pragma: no cover
    
    gambler = get_gambler()
    #gambler.summ_print()
    gambler.layout_print(vname='reward', fmt='', show_env_states=True, none_str=' ')
    gambler.save_to_pickle_file( fname=None )

    gambler.layout.s_hash_diagram( save_name='gambler_diagram', none_str='*', do_show=True,
                                   pad=0.05, scale=0.5, h_over_w=0.8)
    

