
from introrl.layouts.generic_layout import GenericLayout

from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

"""
This maze can grow by a factor, N_mult.
It is used in the prioritized sweep Example 8.4 of Sutton 2nd Ed.

Start remains a single grid location, whereas Goal grows in both x and y.
All Wall entries only grow in both x and y
"""

def get_gridworld(step_reward=0.0, N_mult=1, # N_mult must be an integer.
                  width=9, height=6, goal=(0,8), start=(2,0),
                  wallL=((1,2),(2,2),(3,2),(0,7),(1,7),(2,7),(4,5)) ):

    gridworld = EnvBaseline( name='Sutton Ex8.4 Priority Sweep Maze' ) # GenericLayout set below
    gridworld.set_info( """Sutton Ex8.1 Dyna Maze""" )
    
    width_big = width * N_mult
    height_big = height * N_mult
    
    gridworld.characteristic_dim = width_big + height_big*2
    # get relaxed optimal length from Zhang.
    gridworld.optimal_path_len = int(14 * N_mult * 1.2) + 1
        
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
        
        if j_next >= width_big:
            j_next = j 
        elif j_next < 0:
            j_next = j 
            
        if i_next >= height_big:
            i_next = i
        elif i_next < 0:
            i_next = i
        
        if (i_next,j_next) in wall_set:
            i_next, j_next = i,j 
        
        state_next_hash = (i_next, j_next)
        
        if state_next_hash in goal_set:
            reward = 1.0
        else:
            reward = 0.0
        
        return reward, state_next_hash
    
    def make_big_set( pos ):
        """Take an (i,j) position, pos, and expand to new, big size in x and y"""
        pos_set = set()
        ip, jp = pos
        ip *= N_mult
        jp *= N_mult
        for ixn in range( N_mult ):
            for jxn in range( N_mult ):
                pos_set.add( (ip+ixn, jp+jxn) )
        return pos_set
            
    # define default policy
    gridworld.default_policyD = {} #index=s_hash, value=list of equiprobable actions
    
    # redefine start
    istart, jstart = start
    start = (istart*N_mult, jstart*N_mult)
    
    # make goal set 
    goal_set = make_big_set( goal )
    
    # make wall set 
    wall_set = set()
    for wall in wallL:
        wall_set.update( make_big_set( wall ) )

    # create state hash entries
    for i in range(height_big):
        for j in range(width_big):
            s_hash = (i,j)
            if (s_hash not in wall_set) and (s_hash not in goal_set):
                gridworld.default_policyD[ s_hash ] = ('U','D','R','L')
                for a_desc in ['U','D','R','L']:
                    gridworld.add_action( s_hash, a_desc, a_prob=1.0 ) # a_prob will be normalized

                    reward_val, sn_hash = get_action_snext_reward( s_hash, a_desc )
                    # add each event to transitions object
                    gridworld.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=reward_val)

    gridworld.define_env_states_actions()  # send all states and actions to environment

    # --------------------

    s_hash_rowL = [] # layout rows for makeing 2D output
    for i in range(height_big): # put (0,0) at upper left
        rowL = []
        for j in range(width_big):
            s = (i,j)
            if s in wall_set:
                rowL.append( '"Wall"' )
            else:
                rowL.append( s )
    
        # use insert to put (0,0) at lower left
        s_hash_rowL.append( rowL )# layout rows for makeing 2D output
        
    named_s_hashD = {}
    named_s_hashD[start] = 'Start'
    for g in goal_set:
        named_s_hashD[g] = 'Goal'

    gridworld.layout = GenericLayout( gridworld, s_hash_rowL=s_hash_rowL, 
                                      named_s_hashD=named_s_hashD )


    gridworld.start_state_hash =  start
    
    return gridworld


if __name__ == "__main__": # pragma: no cover
    
    gridworld = get_gridworld( N_mult=1 )
    gridworld.summ_print()
    gridworld.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    #gridworld.save_to_pickle_file( fname=None )
