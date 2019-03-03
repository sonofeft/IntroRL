import sys
import numpy as np
from introrl.mdp_data.simple_grid_world import get_gridworld
from introrl.linear_funcs.baseline_q_func import Baseline_Q_Func
from introrl.agents.sa_semigrad_agent import SA_SemiGradAgent
from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.policy import Policy
from introrl.agent_supt.epsilon_calc import EpsilonGreedy
from introrl.agent_supt.alpha_calc import Alpha

class LazyProgrammerMaze( Baseline_Q_Func ):
    
    def __init__(self, environment ):
        
        Baseline_Q_Func.__init__(self, environment )
        
    def init_w_vector(self):
        """Initialize the weights vector and the number of entries, N."""
        
        # initialize a weights numpy array with random values.
        N = 25
        self.w_vector = np.random.randn(N) / np.sqrt(N)
        self.N = len( self.w_vector )
                
    def get_sa_x_vector(self, s_hash, a_desc):
        """Return the x vector that represents the (s,a) pair."""
        
        s = s_hash
        
        x_vector = np.array([
          s[0] - 1              if a_desc == 'U' else 0,
          s[1] - 1.5            if a_desc == 'U' else 0,
          (s[0]*s[1] - 3)/3     if a_desc == 'U' else 0,
          (s[0]*s[0] - 2)/2     if a_desc == 'U' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a_desc == 'U' else 0,
          1                     if a_desc == 'U' else 0,
          s[0] - 1              if a_desc == 'D' else 0,
          s[1] - 1.5            if a_desc == 'D' else 0,
          (s[0]*s[1] - 3)/3     if a_desc == 'D' else 0,
          (s[0]*s[0] - 2)/2     if a_desc == 'D' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a_desc == 'D' else 0,
          1                     if a_desc == 'D' else 0,
          s[0] - 1              if a_desc == 'L' else 0,
          s[1] - 1.5            if a_desc == 'L' else 0,
          (s[0]*s[1] - 3)/3     if a_desc == 'L' else 0,
          (s[0]*s[0] - 2)/2     if a_desc == 'L' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a_desc == 'L' else 0,
          1                     if a_desc == 'L' else 0,
          s[0] - 1              if a_desc == 'R' else 0,
          s[1] - 1.5            if a_desc == 'R' else 0,
          (s[0]*s[1] - 3)/3     if a_desc == 'R' else 0,
          (s[0]*s[0] - 2)/2     if a_desc == 'R' else 0,
          (s[1]*s[1] - 4.5)/4.5 if a_desc == 'R' else 0,
          1                     if a_desc == 'R' else 0,
          1
        ])
        
        return x_vector

    
    
    
learn_tracker = LearnTracker()
gridworld = get_gridworld( step_reward=-0.1 )

NUM_EPISODES = 2000

alpha_obj = Alpha(alpha=0.1)
alpha_obj.set_half_life_for_N_episodes( Nepisodes=NUM_EPISODES, alpha_final=0.03333333333333)

eps_obj = EpsilonGreedy(epsilon=0.5)
eps_obj.set_half_life_for_N_episodes( Nepisodes=NUM_EPISODES, epsilon_final=0.16666666666666)

agent = SA_SemiGradAgent( environment=gridworld, update_type='qlearn',
                         sa_linear_function=LazyProgrammerMaze( gridworld ),
                         learn_tracker=learn_tracker,
                         gamma=0.9,
                         alpha=alpha_obj,
                         epsilon=eps_obj)


for i in range(NUM_EPISODES):
    agent.run_episode( (2,0))
print()

agent.summ_print()
print('-'*77)
#learn_tracker.summ_print()
#print('-'*77)

agent.action_value_linfunc.summ_print(fmt_Q='%.4f')
print('-'*77)


policy = Policy( environment=gridworld )
for s_hash in gridworld.iter_all_action_states():
    a_desc = agent.action_value_linfunc.get_best_eps_greedy_action( s_hash, epsgreedy_obj=None )
    policy.set_sole_action( s_hash, a_desc)

policy.summ_print( environment=gridworld, verbosity=0 )



