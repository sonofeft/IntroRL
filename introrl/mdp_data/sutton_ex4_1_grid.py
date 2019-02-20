
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward


# --------------------
# define layout for output

s_hash_rowL = []
for i in range(4):# put (0,0) at upper left
    rowL = []
    for j in range(4):
        rowL.append( i*4 + j )
    s_hash_rowL.append( rowL )
rowL[-1] = 0

def get_gridworld(step_reward=-0.04):

    gridworld = EnvBaseline( name='Sutton Ex4.1 Grid World', s_hash_rowL=s_hash_rowL )
    gridworld.set_info( """
        Example 4.1 grid 
        Label for blank space is "0" (both blanks are the same actual state)
        (i.e. upper left corner and lower right corner are state "0")
        """ )

    for state_hash in range(1, 15): # states are numbered 1-14
        for action_desc in ['U','D','R','L']:
            gridworld.add_action( state_hash, action_desc, a_prob=1.0 ) # a_prob will be normalized
            
            a = action_desc
            s = state_hash
            
            if a == 'U':
                sn = s - 4
            elif a == 'D':
                sn = s + 4
            elif a == 'R':
                if s not in [3,7,11]:
                    sn = s + 1
                else:
                    sn = s
            elif a == 'L':
                if s not in [4, 8, 12]:
                    sn = s - 1
                else:
                    sn = s
                    
            if sn < 0:
                sn = s
            elif sn > 15:
                sn = s
            elif sn == 15:
                sn = 0
                
            gridworld.add_transition( state_hash, action_desc, sn, t_prob=1.0, reward_obj=-1.0)
                
    gridworld.define_env_states_actions()  # send all states and actions to environment

    gridworld.start_state_hash =  12


    # define default policy (if any)
    policyD = {} # index=state_hash, value=action_desc

    for (s_hash, a_desc) in gridworld.iter_state_hash_action_desc():
        if s_hash not in policyD:
            policyD[s_hash] = []
        policyD[s_hash].append( (a_desc, 0.25) )
        
    # make policyD entries hashable for later use (i.e. tuple, not list)
    for s_hash, aL in policyD.items():
        policyD[s_hash] = tuple( aL ) 
    
    gridworld.default_policyD = policyD


    return gridworld

if __name__ == "__main__": # pragma: no cover
    
    
    gridworld = get_gridworld()
    #gridworld.summ_print()
    gridworld.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    gridworld.save_to_pickle_file( fname=None )
