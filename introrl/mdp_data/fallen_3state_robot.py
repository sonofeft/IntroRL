
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward

# --------------------
    
s_hash_rowL = [] # layout rows for makeing 2D output

s_hash_rowL.append( ['Standing',  '*',        '*' ])
s_hash_rowL.append( ['*',  '*',   'Moving' ]) 
s_hash_rowL.append( ['Fallen',  '*',        '*'  ])      


def get_robot():

    robot = EnvBaseline( name='Slow-Fast Fallen Robot', s_hash_rowL=s_hash_rowL )
    robot.set_info( """
        Sample 3 State Fallen, Standing, Moving Robot.
        https://sandipanweb.wordpress.com/2017/03/23/some-reinforcement-learning-using-policy-value-iteration-and-q-learning-for-a-markov-decision-process-in-python-and-r/
        Some Reinforcement Learning: Using Policy & Value Iteration and Q-learning for a Markov Decision Process in Python and R
        """ )

    robot.add_action( 'Fallen',   'Slow', a_prob=1.0 )
    robot.add_action( 'Standing', 'Slow', a_prob=1.0 )
    robot.add_action( 'Moving',   'Slow', a_prob=1.0 )

    robot.add_action( 'Standing', 'Fast', a_prob=1.0 )
    robot.add_action( 'Moving',   'Fast', a_prob=1.0 )

    robot.add_transition( 'Fallen',   'Slow', 'Fallen',   t_prob=0.6, reward_obj=-1.0)
    robot.add_transition( 'Fallen',   'Slow', 'Standing', t_prob=0.4, reward_obj=1.0)

    robot.add_transition( 'Standing', 'Slow', 'Moving', t_prob=1.0, reward_obj=1.0)
    robot.add_transition( 'Moving',   'Slow', 'Moving', t_prob=1.0, reward_obj=1.0)

    robot.add_transition( 'Standing', 'Fast', 'Moving', t_prob=0.6, reward_obj=2.0)
    robot.add_transition( 'Standing', 'Fast', 'Fallen', t_prob=0.4, reward_obj=-1.0)

    robot.add_transition( 'Moving', 'Fast', 'Moving', t_prob=0.8, reward_obj=2.0)
    robot.add_transition( 'Moving', 'Fast', 'Fallen', t_prob=0.2, reward_obj=-1.0)

    robot.define_env_states_actions()  # send all states and actions to environment


    robot.start_state_hash = 'Standing'

    # define default policy (if any)
    policyD = {} # index=state_hash, value=action_desc

    policyD[ 'Standing' ] = 'Slow'
    policyD[ 'Fallen' ] = 'Slow'
    policyD[ 'Moving' ] = 'Slow'
    robot.default_policyD = policyD
    
    return robot

if __name__ == "__main__": # pragma: no cover
    
    robot = get_robot()
    robot.summ_print()
    #robot.layout_print(vname='reward', fmt='', show_env_states=True, none_str='*')
    robot.save_to_pickle_file( fname=None )
