from introrl.black_box_sims.random_walk_1000 import RandomWalk_1000Simulation
from introrl.agent_supt.episode_maker import make_episode
from introrl.policy import Policy

NUM_EPISODES = 100000
countD = {} # index=state, value=count 

RW = RandomWalk_1000Simulation()
policy = Policy(environment=RW)
policy.intialize_policy_to_equiprobable( env=RW )


for Nepi in range(NUM_EPISODES):
    episode = make_episode(500, policy, RW, max_steps=10000)
    
    for dr in episode.get_rev_discounted_returns( gamma=1.0 ):
        (s_hash, a_desc, reward, sn_hash, G) = dr
        
        countD[ s_hash ] = countD.get( s_hash, 0 ) + 1

SUM_VISITS = sum( list(countD.values()) )
freqL = []
for i in range(1,1001):
    freqL.append( countD.get(i,0) / float(SUM_VISITS) )

# copy and paste list into plot script
print('freqL =', repr(freqL))
