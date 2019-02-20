
import unittest
# import unittest2 as unittest # for versions of python < 2.7

"""
        Method                            Checks that
self.assertEqual(a, b)                      a == b   
self.assertNotEqual(a, b)                   a != b   
self.assertTrue(x)                          bool(x) is True  
self.assertFalse(x)                         bool(x) is False     
self.assertIs(a, b)                         a is b
self.assertIsNot(a, b)                      a is not b
self.assertIsNone(x)                        x is None 
self.assertIsNotNone(x)                     x is not None 
self.assertIn(a, b)                         a in b
self.assertNotIn(a, b)                      a not in b
self.assertIsInstance(a, b)                 isinstance(a, b)  
self.assertNotIsInstance(a, b)              not isinstance(a, b)  
self.assertAlmostEqual(a, b, places=5)      a within 5 decimal places of b
self.assertNotAlmostEqual(a, b, delta=0.1)  a is not within 0.1 of b
self.assertGreater(a, b)                    a is > b
self.assertGreaterEqual(a, b)               a is >= b
self.assertLess(a, b)                       a is < b
self.assertLessEqual(a, b)                  a is <= b

for expected exceptions, use:

with self.assertRaises(Exception):
    blah...blah...blah

with self.assertRaises(KeyError):
    blah...blah...blah

Test if __name__ == "__main__":
    def test__main__(self):
        # loads and runs the bottom section: if __name__ == "__main__"
        runpy = imp.load_source('__main__', os.path.join(up_one, 'filename.py') )


See:
      https://docs.python.org/2/library/unittest.html
         or
      https://docs.python.org/dev/library/unittest.html
for more assert options
"""

import sys, os

here = os.path.abspath(os.path.dirname(__file__)) # Needed for py.test
up_one = os.path.split( here )[0]  # Needed to find modelps development version
if here not in sys.path[:2]:
    sys.path.insert(0, here)
if up_one not in sys.path[:2]:
    sys.path.insert(0, up_one)
    
from introrl.environments.env_baseline import EnvBaseline
from introrl.layouts.generic_layout import GenericLayout
from introrl.reward import Reward
from introrl.mdp_data.simple_grid_world import get_gridworld
from introrl.mdp_data.six_states import get_six_states

class DummyEnv( EnvBaseline ):
    def __init__(self, name='Dummy Env' ):
        EnvBaseline.__init__(self, name=name)
    def define_environment(self):
        for state_hash in range(1, 25):
            for action_desc in [-1,1]:
                self.add_action( state_hash, action_desc, a_prob=1.0 ) # a_prob will be normalized
                sn = state_hash + action_desc
                self.add_transition( state_hash, action_desc, sn, t_prob=1.0, reward_obj=0.0)
                
        self.define_env_states_actions()  # send all states and actions to environment
        self.start_state_hash =  12
        self.layout = GenericLayout( self )

reward_probL = [(0.0,1), (1.0,1)] # will be normalized in use.
rt = Reward( reward_probL=reward_probL)

class TinyEnv( EnvBaseline ):
    def __init__(self, name='Tiny Env' ):
        EnvBaseline.__init__(self, name=name)
    def define_environment(self):
        for state_hash in range(1, 3):
            for action_desc in [-1,1]:
                self.add_action( state_hash, action_desc, a_prob=1.0 ) # a_prob will be normalized
                sn = state_hash + action_desc
                self.add_transition( state_hash, action_desc, sn, t_prob=1.0, reward_obj=rt)
        self.define_env_states_actions()  # send all states and actions to environment
        self.start_state_hash =  12
        self.layout = GenericLayout( self )

class RogueEnv( EnvBaseline ):
    def __init__(self, name='Tiny Env' ):
        EnvBaseline.__init__(self, name=name)
    def define_environment(self):
        for state_hash in range(1, 3):
            for action_desc in [-1,1]:
                self.add_action( state_hash, action_desc, a_prob=1.0 ) # a_prob will be normalized
                sn = state_hash + action_desc
                self.add_transition( state_hash, action_desc, sn, t_prob=1.0, reward_obj=0.0)
        self.define_env_states_actions()  # send all states and actions to environment
        self.start_state_hash =  12
        self.layout = GenericLayout( self )
    def get_layout_row_col_of_state(self, s_hash ):
        if s_hash in (1,2):
            return (1,1+s_hash)
        return None,None


class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.ENV = get_gridworld()

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.ENV )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""
        
        # See if the self.ENV object exists
        self.assertIsInstance(self.ENV, EnvBaseline, msg=None)

    def test_iterate_action_desc(self):
        """test iterate action desc"""
        aL = [a_desc for a_desc in self.ENV.iter_action_desc_prob( (0,0) , incl_zero_prob=False)]
        self.assertEqual(len(aL), 2)

        self.assertIn(('D', 0.5), aL)
        self.assertIn(('R', 0.5), aL)

    def test_iterate_next_state(self):
        """test iterate next state"""
        aL = [a_desc for a_desc in 
              self.ENV.iter_next_state_prob_reward( (0,0), 'R', incl_zero_prob=False)]
        self.assertEqual(len(aL), 1)
        self.assertIn(((0, 1), 1.0, 0.0), aL)
        
    def test_integer_get_layout_row_col(self):
        """test integer get layout row col"""
        self.ENV.add_action( (0,0), 'D', a_prob=1.0)
        self.ENV.add_action( (0,0), 'L', a_prob=1.0)
        self.ENV.add_transition( (0,0), 'D', 10, t_prob=1.0, reward_obj=0.0)
        self.ENV.add_transition( (0,0), 'L', 22, t_prob=1.0, reward_obj=0.0)
        self.ENV.define_env_states_actions()
        
        (row, col) = self.ENV.get_layout_row_col_of_state( 10 )
        self.assertEqual((row, col), (2,3))
        
        (row, col) = self.ENV.get_layout_row_col_of_state( 22 )
        self.assertEqual((row, col), (3,0))
        
        
    def test_integer_get_layout_row_col_v2(self):
        """test integer get layout row col"""
        dummyenv = DummyEnv()
        (row, col) = dummyenv.get_layout_row_col_of_state( 10 )
        self.assertEqual((row, col), (1,4))
        
        
        
    def test_get_est_reward(self):
        """test get est rewards"""
        
        est_rD, msgD = self.ENV.get_estimated_rewards()
        self.assertEqual(est_rD[(0,0)], 0.0)
        self.assertEqual(msgD, {})

        tinyenv = TinyEnv()
        est_rD, msgD = tinyenv.get_estimated_rewards()
        self.assertEqual(msgD, {0: 'est', 2: 'est', 1: 'est', 3: 'est'})

        # test GenericLayout unrecognized s_hash
        #tinyenv.add_action( 1, 'U', a_prob=1.0)
        #tinyenv.add_transition( 1, 'U', 'X', t_prob=1.0, reward_obj=0.0)
        #tinyenv.layout_print( vname='reward', fmt='%6g', show_env_states=False, none_str='*')

    def test_summ_print(self):
        """test summ_print"""
        tinyenv = TinyEnv()
        tinyenv.summ_print()        
        
        # test bail-out logic
        tinyenv.layout = None
        tinyenv.layout_print( vname='reward', fmt='%6g', show_env_states=False, none_str='*')

        # grab summ str
        s = tinyenv.SAC.get_state_summ_str( 1 )
        self.assertEqual(s, '1(0.5)-1(0.5)' )

    def test_misfit_layout_s_hash(self):
        """test misfit layout s_hash"""
        rogueenv = RogueEnv()
        print('rogueenv=',rogueenv)
        #rogueenv.summ_print()
        (x,y) = rogueenv.layout.get_s_hash_xy( 1 )
        self.assertEqual((x,y), (1,2))
        
        (x,y) = rogueenv.layout.get_s_hash_xy( 0 )
        self.assertEqual((x,y), (2,0))
        


    def test_define_layout_w_s_hash_rowL(self):
        """test define layout w s_hash_rowL"""
        
        env = get_six_states()
        pD = env.get_default_policy_desc_dict()
        
        # check next state
        sn_hash = env.TC.get_prob_weighted_next_state_hash( 'A', 'ur')
        self.assertEqual(sn_hash, None)
        sn_hash = env.TC.get_prob_weighted_next_state_hash( 'A', 'U')
        self.assertEqual(sn_hash, 'B')
        
        # check reward
        rval = env.TC.get_reward_value('A', 'U', 'B')
        self.assertEqual(rval, 0.0)

        rval = env.TC.get_reward_value('A', 'U', 'ur')
        self.assertEqual(rval, None)
        
        rval = env.TC.get_reward_value('A', 'ur', 'B')
        self.assertEqual(rval, None)

        rval = env.TC.get_reward_value('X', 'U', 'B')
        self.assertEqual(rval, None)


if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

