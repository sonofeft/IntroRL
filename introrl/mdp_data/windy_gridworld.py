
from introrl.layouts.generic_layout import GenericLayout

from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

def get_gridworld(step_reward=-1.0, height=7, goal=(3,7), start=(3,0),
            windT=(0,0,0,1,1,1,2,2,1,0)):
    """
    Windy Gridworld with (0,0) at lower left
    width is defined by length of windT tuple.
    """

    gridworld = EnvBaseline( name='Windy Gridworld Sutton Ex6.5' ) # GenericLayout set below
    gridworld.set_info( """""" )

    width = len( windT )

    def get_action_snext( s_hash, action):
        """returns state_next_hash"""
        
        if s_hash == 'Start':
            s_hash = start

        di = 0
        dj = 0

        if action=='U':
            di = 1
        elif action=='D':
            di = -1
        elif action=='R':
            dj = 1
        elif action=='L':
            dj = -1

        (i,j) = s_hash
        wind_di = windT[ j ]

        i_next = i + di
        # constrain basic move to be inside the grid
        i_next = max(0, min(height-1, i_next))

        i_next += wind_di # add wind to constrained move.
        j_next = j + dj

        # constrain next position to be inside the grid
        i_next = max(0, min(height-1, i_next))
        j_next = max(0, min(width-1, j_next))

        state_next_hash = (i_next, j_next)
        if state_next_hash == goal:
            state_next_hash = 'Goal'
        if state_next_hash == start:
            state_next_hash = 'Start'
        return state_next_hash


    # define default policy
    gridworld.default_policyD = {} #index=s_hash, value=list of equiprobable actions

    for i in range(height):
        for j in range(width):
            s_hash = (i,j)
            if s_hash == start:
                s_hash = 'Start'
            
            if s_hash == goal:
                pass  # s_hash == 'Goal'
            else:
                gridworld.default_policyD[ s_hash ] = ('U','D','R','L')
                for a_desc in ['U','D','R','L']:
                    gridworld.add_action( s_hash, a_desc, a_prob=1.0 ) # a_prob will be normalized

                    sn_hash = get_action_snext( s_hash, a_desc )
                    # add each event to transitions object
                    gridworld.add_transition( s_hash, a_desc, sn_hash, t_prob=1.0, reward_obj=step_reward)

    gridworld.define_env_states_actions()  # send all states and actions to environment

    # --------------------

    s_hash_rowL = [] # layout rows for makeing 2D output
    for i in range(height):
        rowL = []
        for j in range(width):
            s_hash = (i,j)
            if s_hash == goal:
                s_hash = 'Goal'
            elif s_hash == start:
                s_hash = 'Start'

            rowL.append( s_hash )

        # use insert to put (0,0) at lower left, append for upper left
        s_hash_rowL.insert(0, rowL )# layout rows for makeing 2D output

    gridworld.layout = GenericLayout( gridworld, s_hash_rowL=s_hash_rowL,
                                      col_tickL=windT,
                                      x_axis_label='Upward Wind Speed'  )


    gridworld.start_state_hash =  'Start'

    return gridworld


if __name__ == "__main__": # pragma: no cover

    gridworld = get_gridworld()
    if 0:
        gridworld.summ_print()
    else:
        gridworld.layout_print(vname='reward', fmt='%.2f', show_env_states=True, none_str='*')
    gridworld.save_to_pickle_file( fname=None )
