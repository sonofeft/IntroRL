
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

# --------------------

s_hash_rowL = [] # layout rows for makeing 2D output
for i in range(5): # put (0,0) at upper left
    rowL = []
    for j in range(5):
        rowL.append( (i,j) )

    # use insert to put (0,0) at lower left
    s_hash_rowL.insert(0, rowL )# layout rows for makeing 2D output


def get_gridworld(step_reward=-0.04):

    gridworld = EnvBaseline( name='Sutton Ex4.1 5x5 Grid World', s_hash_rowL=s_hash_rowL )
    gridworld.set_info( """
           Sutton 5x5 Gridworld
        Book Answer from page 65 (linear eqn solve) for gamma=0.9
         22.0     24.4      22.0      19.4      17.5
         19.8     22.0      19.8      17.8      16.0
         17.8     19.8      17.8      16.0      14.4
         16.0     17.8      16.0      14.4      13.0
         14.4     16.0      14.4      13.0      11.7
    =================================================    """ )

        
    def get_action_snext_reward( s_hash, action):
        """returns reward and state_next_hash"""
        
        di = 0
        dj = 0
        reward = 0
        
        if action=='N':
            di = 1
        elif action=='S':
            di = -1
        elif action=='E':
            dj = 1
        elif action=='W':
            dj = -1
        
        (i,j) = s_hash
        i_next = i + di
        j_next = j + dj
        
        if (i==4) and (j==1):
            i_next = 0
            j_next = 1
            reward = 10
        elif (i==4) and (j==3):
            i_next = 2
            j_next = 3
            reward = 5
        elif (i_next<0) or (i_next>4) or (j_next<0) or (j_next>4):
            i_next = i
            j_next = j
            reward = -1
        
        state_next_hash = (i_next, j_next)
        return reward, state_next_hash
        
            
    # define default policy
    gridworld.default_policyD = {} #index=s_hash, value=list of equiprobable actions

    for i in range(5):
        for j in range(5):
            s_hash = (i,j)
            
            gridworld.default_policyD[ s_hash ] = ('N','S','E','W')
            for a_desc in ['N','S','E','W']:
                gridworld.add_action( s_hash, a_desc, a_prob=1.0 ) # a_prob will be normalized

                reward_val, sn_hash = get_action_snext_reward( s_hash, a_desc )
                # add each event to transitions object
                gridworld.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=reward_val)

    gridworld.define_env_states_actions()  # send all states and actions to environment


    gridworld.start_state_hash =  (0,0)
    
    return gridworld


if __name__ == "__main__": # pragma: no cover
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    
    gridworld = get_gridworld()
    #gridworld.summ_print()
    gridworld.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    gridworld.save_to_pickle_file( fname=None )



    policy, state_value = dp_value_iteration( gridworld, do_summ_print=True,
                                              max_iter=1000, err_delta=0.001, 
                                              gamma=0.9)
                                              
    policy.save_diagram( gridworld, inp_colorD=None, save_name='sutton_5x5_gridworld',
                         show_arrows=True, scale=1.0, h_over_w=0.8)

