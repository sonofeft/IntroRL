import sys
import numpy as np

from introrl.linear_funcs.baseline_v_func import Baseline_V_Func
from introrl.black_box_sims.random_walk_1000 import RandomWalk_1000Simulation
from introrl.utils.tiles_rectangles import PartitionedSegment
from introrl.utils import pickle_esp
from math import sqrt

_, STATE_VALUES, _ = pickle_esp.read_pickle_file( fname='random_walk_1000_PI_eval')

class AggregatedRandomWalk1000_Func( Baseline_V_Func ):
    
    def __init__(self, environment,  num_regions=10):
        
        self.pseg = PartitionedSegment( lo_val=0, hi_val=1000, num_regions=num_regions )
        self.num_regions = num_regions
        
        Baseline_V_Func.__init__(self, environment )
        
        
        
    def init_w_vector(self):
        """Initialize the weights vector and the number of entries, N."""
        # initialize a weights numpy array with random values.
        self.w_vector = np.zeros( self.num_regions )
        self.N = len( self.w_vector )
                
    def get_x_vector(self, s_hash):
        """Return the x vector that represents the state."""
        
        return self.pseg.get_numpy_encoding( s_hash )

    def calc_rms_error( self ):
        """Using the dictionary, STATE_VALUES.VsD as reference, calc RMS error."""
        diff_sqL = []
        for s_hash, true_val in STATE_VALUES.VsD.items():
            diff_sqL.append( (true_val - self.VsEst(s_hash))**2 )
        rms = sqrt( sum( diff_sqL ) / len(diff_sqL) )
        return rms


if __name__ == "__main__": # pragma: no cover
    from introrl.agent_supt.learning_tracker import LearnTracker
    from introrl.policy import Policy
    from introrl.agent_supt.epsilon_calc import EpsilonGreedy
    from introrl.agent_supt.alpha_calc import Alpha
    from introrl.agent_supt.episode_maker import make_episode
    from introrl.agent_supt.nstep_td_eval_walker import NStepTDWalker

    RW = RandomWalk_1000Simulation()
    ARWfunc = AggregatedRandomWalk1000_Func( RW, num_regions=20)
    policy = Policy(environment=RW)
    policy.intialize_policy_to_equiprobable( env=RW )
    
    episode = make_episode(500, policy, RW, max_steps=10000)
    
    NSwalker = NStepTDWalker(RW, Nsteps=4,  episode_obj=episode)
    
    
    for i in range(10):
        NSwalker.set_episode_obj( episode )
        NSwalker.do_td_sv_function_updates( ARWfunc, alpha=0.1, gamma=1.0 )
        episode = make_episode(500, policy, RW, max_steps=10000)
    
    print( ARWfunc.w_vector )
    print('RMS =', ARWfunc.calc_rms_error() )
    
