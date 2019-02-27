
from introrl.layouts.generic_layout import GenericLayout
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

def get_gridworld( step_reward=0.0 ):
    gridworld = EnvBaseline( name='Simple Grid World' ) # GenericLayout set below
    gridworld.set_info( 'Simple Grid World Example.' )

    actionD = {(0, 0): ('D', 'R'),
               (0, 1): ('L', 'R'),
               (0, 2): ('L', 'D', 'R'),
               (1, 0): ('U', 'D'),
               (1, 2): ('U', 'D', 'R'),
               (2, 0): ('U', 'R'),
               (2, 1): ('L', 'R'),
               (2, 2): ('L', 'R', 'U'),
               (2, 3): ('L', 'U')  }
                   
    rewardD = {(0, 3): 1, (1, 3): -1}

    for state_hash, actionL in actionD.items():
        
        for action_desc in actionL:
            gridworld.add_action( state_hash, action_desc, a_prob=1.0 ) # a_prob will be normalized
            
            a = action_desc
            s = state_hash
            
            if a == 'U':
                state_next_hash = (s[0]-1, s[1])
            elif a == 'D':
                state_next_hash = (s[0]+1, s[1])
            elif a == 'R':
                state_next_hash = (s[0], s[1]+1)
            elif a == 'L':
                state_next_hash = (s[0], s[1]-1)

            reward_val = rewardD.get( state_next_hash, step_reward )
            
            gridworld.add_transition( state_hash, action_desc, state_next_hash, t_prob=1.0, 
                                      reward_obj=reward_val)
            
    gridworld.define_env_states_actions()  # send all states and actions to environment

    gridworld.layout = GenericLayout( gridworld )# uses default "get_layout_row_col_of_state"

    # If there is a start state, define it here.
    gridworld.start_state_hash = (2,0)
    gridworld.define_limited_start_state_list( [(2,0), (2,2)] )

    # define default policy (if any)
    # Policy Dictionary for: GridWorld

    policyD = {} # index=state_hash, value=action_desc

    #                 Vpi shown for gamma=0.9
    policyD[(0, 0)] = 'R'    # Vpi=0.81
    policyD[(1, 0)] = 'U'    # Vpi=0.729
    policyD[(0, 1)] = 'R'    # Vpi=0.9
    policyD[(0, 2)] = 'R'    # Vpi=1.0
    policyD[(1, 2)] = 'U'    # Vpi=0.9
    policyD[(2, 0)] = 'U'    # Vpi=0.6561
    policyD[(2, 2)] = 'U'    # Vpi=0.81
    policyD[(2, 1)] = 'R'    # Vpi=0.729
    policyD[(2, 3)] = 'L'    # Vpi=0.729

    gridworld.default_policyD = policyD
    
    return gridworld

if __name__ == "__main__": # pragma: no cover
    
    gridworld = get_gridworld()
    score = gridworld.get_policy_score( gridworld.get_default_policy_desc_dict(), step_limit=1000)
    print('Policy Score =', score, ' = (r_sum, n_steps, msg)')
    
    #gridworld.summ_print()
    gridworld.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    
    gridworld.save_to_pickle_file( fname=None )
