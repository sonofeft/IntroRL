
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

from introrl.policy import Policy
from introrl.mdp_data.simple_grid_world import get_gridworld

class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.gridworld = get_gridworld()
        self.P = Policy(  environment=self.gridworld  )
        self.P.intialize_policy_to_equiprobable( env=self.gridworld )

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.P )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""
        
        # See if the self.P object exists
        self.assertIsInstance(self.P, Policy, msg=None)
        
    def test_set_policy_from_default_pi(self):
        """test set policy from default pi"""
        
        policyD = self.gridworld.get_default_policy_desc_dict()
        self.P.set_policy_from_piD( policyD )

        self.assertEqual(self.P.get_action_prob( (2,2), 'U'), 1.0)
        self.assertEqual(self.P.get_action_prob( (2,2), 'R'), 0.0)
        self.assertEqual(self.P.get_action_prob( (2,2), 'D'), None)

    #def test_set_policy_from_list_of_actions(self):
    #    """test set policy from list of actions"""
    #    piD = {(0, 0):('R','D') }
    #    self.P.set_policy_from_piD( piD )

    #    self.assertEqual(self.P.get_action_prob( (0,0), 'U'), None)
    #    self.assertEqual(self.P.get_action_prob( (0,0), 'R'), 0.5)
    #    self.assertEqual(self.P.get_action_prob( (0,0), 'D'), 0.5)

    #def test_set_policy_from_list_of_action_probs(self):
    #    """test set policy from list of action probs"""
    #    piD = {(0, 0):[('R',0.6), ('D',0.4)] }
    #    self.P.set_policy_from_piD( piD )

    #    self.assertEqual(self.P.get_action_prob( (0,0), 'U'), None)
    #    self.assertEqual(self.P.get_action_prob( (0,0), 'R'), 0.6)
    #    self.assertEqual(self.P.get_action_prob( (0,0), 'D'), 0.4)
            
    #    # make (action, prob) entry too long.
    #    with self.assertRaises(ValueError):
    #        piD = {(0, 0):[('R',0.6,0.4), ('D',0.4,0.6)] }
    #        self.P.set_policy_from_piD( piD )

    def test_learn_all_s_and_a(self):
        """test learn all s and a"""

        self.P.learn_all_states_and_actions_from_env( self.gridworld )

    def test_initialize_to_random(self):
        """test initialize to random"""

        self.P.intialize_policy_to_random( env=self.gridworld )
        apL = self.P.get_list_of_all_action_desc_prob( (0,2), incl_zero_prob=True)
        pL = [p for (adesc,p) in apL]
        self.assertEqual( sorted(pL), [ 0.0, 0.0, 1.0] )

    def test_iterate_adesc_p(self):
        """test iterate adesc p"""
        
        apL = []
        for (a_desc,p) in self.P.iter_policy_ap_for_state( (0,0), incl_zero_prob=False):
            apL.append( (a_desc,p) )
                
        self.assertIn(('R',0.5), apL)
        self.assertIn(('D',0.5), apL)
        self.assertNotIn(('U',0.5), apL)

    def test_iterate_all_states(self):
        """test iterate all states"""
        
        sL = []
        for s_hash in self.P.iter_all_policy_states():
            sL.append( s_hash )
        sL.sort()
        self.assertEqual(len(sL), 9)
        self.assertEqual(sL[0], (0,0))
        self.assertEqual(sL[-1], (2,3))

    def test_get_single_action(self):
        """test get single action"""
        a_desc = self.P.get_single_action( (0,0) )
        self.assertIn(a_desc, ('R','D'))

        a_desc = self.P.get_single_action( (99,99) )
        self.assertEqual(a_desc, None)

if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

