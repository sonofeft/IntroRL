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

alpha = 2.0E-5

pseg = PartitionedSegment( lo_val=0, hi_val=1000, num_regions=10 )
#pseg.summ_print()

def get_x_vector( state ):
    """Return the x vector that represents the (s,a) pair."""
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

#w_vector = np.zeros( pseg.num_regions )
w_vector = np.random.random_sample( pseg.num_regions )
gamma = 1.0

for Nepi in range(NUM_EPISODES):
    episode = make_episode(500, policy, RW, max_steps=10000)
    
    alpha = alpha_obj()
    alpha_obj.inc_N_episodes()

    #episode.summ_print()
    for dr in episode.iter_all_sars():
        
        (s_hash, a_desc, reward, sn_hash) = dr
        Vs    = VsEst( s_hash )
        
        if sn_hash in RW.terminal_set:
            target_val = reward
        else:
            Vstp1 = VsEst( sn_hash )
            target_val = reward + gamma*Vstp1
            
        delta = alpha * (target_val - Vs)
        
        delta_vector = delta * get_gradient( s_hash )
        w_vector += delta_vector
        
    
print('w_vector =', repr(w_vector) )
print()
alpha_obj.summ_print()