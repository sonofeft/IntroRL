import numpy as np

from introrl.black_box_sims.random_walk_1000 import RandomWalk_1000Simulation
from introrl.agent_supt.episode_maker import make_episode
from introrl.policy import Policy
from introrl.utils.tiles_rectangles import PartitionedSegment
from introrl.agent_supt.alpha_calc import Alpha

NUM_EPISODES = 100000

RW = RandomWalk_1000Simulation()
policy = Policy(environment=RW)
policy.intialize_policy_to_equiprobable( env=RW )

alpha_obj = Alpha(alpha=0.01)
alpha_obj.set_half_life_for_N_episodes( Nepisodes=NUM_EPISODES, alpha_final=2.0E-5)

pseg = PartitionedSegment( lo_val=0, hi_val=1000, num_regions=10 )
#pseg.summ_print()

def get_x_vector( state ):
    """Return the x vector that represents the state."""
    x_vector = pseg.get_numpy_encoding( state )
    return x_vector

def VsEst( state ):
    """Return the current estimate for V(s) from linear function eval."""
    x_vector = get_x_vector( state )
    return w_vector.dot( x_vector )

def get_gradient( state ):
    """
    Return the gradient of value function with respect to w_vector.
    Since the function is linear in w, the gradient is = x_vector.
    """
    return get_x_vector( state )

# Could initialize randomly or all zeros.
#w_vector = np.zeros( pseg.num_regions )
w_vector = np.random.random_sample( pseg.num_regions )

for Nepi in range(NUM_EPISODES):
    episode = make_episode(500, policy, RW, max_steps=10000)
    
    alpha = alpha_obj()
    alpha_obj.inc_N_episodes()

    #episode.summ_print()
    for dr in episode.get_rev_discounted_returns( gamma=1.0 ):
        (s_hash, a_desc, reward, sn_hash, G) = dr
        #print(s_hash, G, pseg.get_numpy_encoding( s_hash ))
        
        Vs    = VsEst( s_hash )
        delta = alpha * (G - Vs)
        
        delta_vector = delta * get_gradient( s_hash )
        w_vector += delta_vector

# copy and paste w_vector array into plot script
print('w_vector =',w_vector)
print()
alpha_obj.summ_print()