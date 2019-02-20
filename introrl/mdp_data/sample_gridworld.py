
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

# define layout to create output displays
row_1 = [ (0,0), (0,1),   (0,2), 'Goal' ]
row_2 = [ (1,0),'"Wall"', (1,2), 'Pit' ]
row_3 = [ 'Start', (2,1),   (2,2), (2,3) ]
s_hash_rowL=[row_1, row_2, row_3]

# add layout row and column markings (if any)
row_tickL=[ 0, 1, 2]
col_tickL=[ 0, 1, 2, 3]
x_axis_label='cols'
y_axis_label='rows'

# one way to define actions is an explicit dict of actions.
# (can also simply provide logic within a function to define actions)
actionD = {(0, 0): ('D', 'R'),
           (0, 1): ('L', 'R'),
           (0, 2): ('L', 'D', 'R'),
           (1, 0): ('U', 'D'),
           (1, 2): ('U', 'D', 'R'),
           'Start': ('U', 'R'),
           (2, 1): ('L', 'R'),
           (2, 2): ('L', 'R', 'U'),
           (2, 3): ('L', 'U')  }

# define rewards
rewardD = {'Goal': 1, 'Pit': -1}

def get_next_state( s_hash, a_desc ):
    """use layout definition to get next state"""
    
    if s_hash == 'Start':
        s_hash = (2,0)
    row,col = s_hash # all non-terminal s_hash are (row, col)
    if a_desc == 'U':
        row -= 1
    elif a_desc == 'D':
        row += 1
    elif a_desc == 'R':
        col += 1
    elif a_desc == 'L':
        col -= 1
    # no limit checking done... assume only legal moves are submitted
    return s_hash_rowL[row][col]

def get_gridworld():
    gridworld = EnvBaseline( name='Sample Grid World',s_hash_rowL=s_hash_rowL, 
                             row_tickL=row_tickL, x_axis_label=x_axis_label, 
                             col_tickL=col_tickL, y_axis_label=y_axis_label,
                             colorD={'Goal':'g', 'Pit':'r', 'Start':'b'},
                             basic_color='skyblue')
                             
    gridworld.set_info( 'Sample Grid World showing basic MDP creation.' )

    # add actions from each state 
    #   (note: a_prob will be normalized within add_action_dict)
    gridworld.add_action_dict( actionD )
    
    # for each action, define the next state and transition probability 
    # (here we use the layout definition to aid the logic)
    for s_hash, aL in actionD.items():
        for a_desc in aL:
            sn_hash = get_next_state( s_hash, a_desc )
            reward = rewardD.get( sn_hash, 0.0 )
            
            # for deterministic MDP, use t_prob=1.0
            gridworld.add_transition( s_hash, a_desc, sn_hash, 
                                      t_prob=1.0, reward_obj=reward)
    
    # after the "add" commands, send all states and actions to environment
    # (any required normalization is done here as well.)
    gridworld.define_env_states_actions()  

    # If there is a start state, define it here.
    gridworld.start_state_hash = 'Start'
    
    # If a limited number of start states are desired, define them here.
    gridworld.define_limited_start_state_list( [(2,0), (2,2)] )

    # if a default policy is desired, define it as a dict.
    gridworld.default_policyD = {(0, 0):'R',(1, 0):'U',(0, 1):'R',(0, 2):'R',
                                 (1, 2):'U','Start':'U',(2, 2):'U',(2, 1):'R',
                                 (2, 3):'L'}

    return gridworld

if __name__ == "__main__": # pragma: no cover
    
    gridworld = get_gridworld()
    #gridworld.summ_print()
    gridworld.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')

    gridworld.layout.s_hash_diagram( save_name='sample_diagram', 
                                     none_str='*', do_show=True,
                                     pad=0.05, scale=0.75, h_over_w=1.0)

    gridworld.save_to_pickle_file( fname=None )
