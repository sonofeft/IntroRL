
from introrl.layouts.generic_layout import GenericLayout

from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

def get_gridworld(step_reward=0.0, width=9, height=6, goal=(0,8), start=(2,0),
            wallL=((1,2),(2,2),(3,2),(0,7),(1,7),(2,7),(4,5)) ):

    gridworld = EnvBaseline( name='Sutton Ex8.1 Dyna Maze' ) # GenericLayout set below
    gridworld.set_info( """Sutton Ex8.1 Dyna Maze""" )
        
    def get_action_snext_reward( s_hash, action):
        """returns reward and state_next_hash"""
        
        di = 0
        dj = 0
        reward = 0
        
        if action=='U':
            di = -1
        elif action=='D':
            di = 1
        elif action=='R':
            dj = 1
        elif action=='L':
            dj = -1
        
        (i,j) = s_hash
        i_next = i + di
        j_next = j + dj
        
        if j_next >= width:
            j_next = j 
        elif j_next < 0:
            j_next = j 
            
        if i_next >= height:
            i_next = i
        elif i_next < 0:
            i_next = i
        
        if (i_next,j_next) in wallL:
            i_next, j_next = i,j 
        
        state_next_hash = (i_next, j_next)
        
        if state_next_hash == goal:
            reward = 1.0
        else:
            reward = 0.0
        
        return reward, state_next_hash
        
            
    # define default policy
    gridworld.default_policyD = {} #index=s_hash, value=list of equiprobable actions

    for i in range(height):
        for j in range(width):
            s_hash = (i,j)
            if s_hash != goal:
                gridworld.default_policyD[ s_hash ] = ('U','D','R','L')
                for a_desc in ['U','D','R','L']:
                    gridworld.add_action( s_hash, a_desc, a_prob=1.0 ) # a_prob will be normalized

                    reward_val, sn_hash = get_action_snext_reward( s_hash, a_desc )
                    # add each event to transitions object
                    gridworld.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=reward_val)

    gridworld.define_env_states_actions()  # send all states and actions to environment

    # --------------------

    s_hash_rowL = [] # layout rows for makeing 2D output
    for i in range(height): # put (0,0) at upper left
        rowL = []
        for j in range(width):
            s = (i,j)
            if s in wallL:
                rowL.append( '"Wall"' )
            else:
                rowL.append( s )
    
        # use insert to put (0,0) at lower left
        s_hash_rowL.append( rowL )# layout rows for makeing 2D output

    gridworld.layout = GenericLayout( gridworld, s_hash_rowL=s_hash_rowL  )


    gridworld.start_state_hash =  start
    
    return gridworld


if __name__ == "__main__": # pragma: no cover
    
    gridworld = get_gridworld()
    gridworld.summ_print()
    gridworld.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    gridworld.save_to_pickle_file( fname=None )
