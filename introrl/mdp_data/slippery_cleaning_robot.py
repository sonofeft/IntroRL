
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward
from introrl.utils.functions import clamp

# -----------------------------------------------------------------

s_hash_rowL = [] # layout rows for makeing 2D output
for i in range(1, 4):
    rowL = []
    for j in range(1, 5):
        rowL.append( (i,j) )

    # use insert to put origin at lower left
    s_hash_rowL.insert(0, rowL )# layout rows for makeing 2D output


def get_robot(step_reward=-0.04):

    gridworld = EnvBaseline( name='Slipper Cleaning Robot', s_hash_rowL=s_hash_rowL )
    gridworld.set_info( """
        Example taken from "Dissecting Reinforcement Learning-Part 1" 
        Dec 9, 2016   Massimiliano Patacchiola
        https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
        """ )



    def get_right_angle_list(a):
        
        if a == 'U':
            raL = ['L','R']
        elif a == 'D':
            raL = ['L','R']
        elif a == 'R':
            raL = ['U','D']
        elif a == 'L':
            raL = ['U','D']
            
        return raL

    def get_move_s_next(a, s):
        
        sn = s
        if a == 'U':
            sn = (s[0]+1, s[1])
        elif a == 'D':
            sn = (s[0]-1, s[1])
        elif a == 'R':
            sn = (s[0], s[1]+1)
        elif a == 'L':
            sn = (s[0], s[1]-1)
        
        if sn==(2,2):# can't move into block in the middle.
            sn = s
        
        # limit moves to inside the edges.
        sn_hash = ( clamp(sn[0], 1,3), clamp(sn[1], 1,4) )
        
        return sn_hash
        

    non_termL = [(3, 1),(3, 2),(3, 3),(2, 1),(2, 3),(1, 1),(1, 2),(1, 3),(1, 4)]

    rewardD = {(3, 4): 1, (2, 4): -1}


    # put in 80% and both 10% moves to target
    for s_hash in non_termL:
        for a_desc in ['U','D','L','R']: # normal move
            gridworld.add_action( s_hash, a_desc, a_prob=0.25 )
            
            # 80%
            sn_hash = get_move_s_next(a_desc, s_hash)
            reward_val = rewardD.get( sn_hash, step_reward )
            
            gridworld.add_transition( s_hash, a_desc, sn_hash, t_prob=0.8, reward_obj=reward_val)
            
            # both 10%
            right_angL = get_right_angle_list( a_desc )
            for ar_desc in right_angL:
                sn_hash = get_move_s_next(ar_desc, s_hash)
                reward_val = rewardD.get( sn_hash, step_reward )

                gridworld.add_transition( s_hash, a_desc, sn_hash, t_prob=0.8, reward_obj=reward_val)
    gridworld.define_env_states_actions()

    # If there is a start state, define it here.
    gridworld.start_state_hash = (1,1)

    # define default policy (if any)
    policyD = {} # index=s_hash, value=a_desc

    policyD[(3, 1)] = 'R'
    policyD[(3, 3)] = 'R'
    policyD[(3, 2)] = 'R'

    policyD[(2, 1)] = 'U'
    policyD[(2, 3)] = 'U'

    policyD[(1, 1)] = 'U'
    policyD[(1, 2)] = 'L'
    policyD[(1, 3)] = 'L'
    policyD[(1, 4)] = 'L'

    gridworld.default_policyD = policyD
    
    return gridworld

if __name__ == "__main__": # pragma: no cover
    
    gridworld = get_robot()
    #gridworld.summ_print()
    gridworld.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    gridworld.save_to_pickle_file( fname=None )
