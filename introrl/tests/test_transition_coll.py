
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

import random
from introrl.transition_coll import TransitionColl
from introrl.reward import Reward

def my_gauss():
    return random.gauss(3.0, 0.5)


class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.TC = TransitionColl()
        
            
        actionD = {(0, 0): ('D', 'R'),
                   (0, 1): ('L', 'R'),
                   (0, 2): ('L', 'D', 'R'),
                   (1, 0): ('U', 'D'),
                   (1, 2): ('U', 'D', 'R'),
                   (2, 0): ('U', 'R'),
                   (2, 1): ('L', 'R'),
                   (2, 2): ('L', 'R', 'U'),
                   (2, 3): ('L', 'U')  }
                       
        rewardD = {(0, 3): 1, (1, 3): -1}
        

        for state_hash, actionL in actionD.items():
            for action_desc in actionL:
                
                a = action_desc
                s = state_hash
                
                if a == 'U':
                    snext_hash = (s[0]-1, s[1])
                elif a == 'D':
                    snext_hash = (s[0]+1, s[1])
                elif a == 'R':
                    snext_hash = (s[0], s[1]+1)
                elif a == 'L':
                    snext_hash = (s[0], s[1]-1)

                reward_val = rewardD.get( snext_hash, 0.0 )
                
                self.TC.set_transition( s, a,
                                        snext_hash, reward_obj=Reward(const=reward_val), 
                                        action_prob=1.0, trans_prob=1.0)
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.TC )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""
        
        # See if the self.TC object exists
        self.assertIsInstance(self.TC, TransitionColl, msg=None)
        
        self.assertEqual( len(self.TC), 21)

    def test_set_transition_prob(self):
        """test set transition prob"""
        # create a second possibility for (0,0), 'R'
        self.TC.set_transition( (0,0), 'R',
                                (0,3), reward_obj=Reward(const=1.0), 
                                action_prob=1.0, trans_prob=1.0)

        Sn1,p1 = self.TC.get_next_state_prob( (0,0), 'R', (0,1))
        Sn2,p2 = self.TC.get_next_state_prob( (0,0), 'R', (0,3))
        self.assertEqual(p1, 0.5)
        self.assertEqual(p1, p2)
        self.assertNotEqual(Sn1, Sn2)
        
        # set explicitly
        self.TC.set_transition_prob( (0,0), 'R', (0,1), prob=0.1)
        self.TC.set_transition_prob( (0,0), 'R', (0,3), prob=0.9)
        
        Sn1,p1 = self.TC.get_next_state_prob( (0,0), 'R', (0,1))
        Sn2,p2 = self.TC.get_next_state_prob( (0,0), 'R', (0,3))
        self.assertEqual(p1, 0.1)
        self.assertEqual(p2, 0.9)
        
        # try setting sole prob
        self.TC.set_sole_transition( (0,0), 'R', (0,1) )
        Sn1,p1 = self.TC.get_next_state_prob( (0,0), 'R', (0,1))
        Sn2,p2 = self.TC.get_next_state_prob( (0,0), 'R', (0,3))
        self.assertEqual(p1, 1.0)
        self.assertEqual(p2, 0.0)
        
        # try sole random
        self.TC.initialize_sole_random( (0,0), 'R' )
        Sn1,p1 = self.TC.get_next_state_prob( (0,0), 'R', (0,1))
        Sn2,p2 = self.TC.get_next_state_prob( (0,0), 'R', (0,3))
        pL = sorted( [p1, p2] )
        self.assertEqual(pL, [0.0, 1.0])
        
        # try equiprobable
        self.TC.intialize_to_equiprobable( (0,0), 'R', )
        Sn1,p1 = self.TC.get_next_state_prob( (0,0), 'R', (0,1))
        Sn2,p2 = self.TC.get_next_state_prob( (0,0), 'R', (0,3))
        self.assertEqual(p1, 0.5)
        self.assertEqual(p1, p2)
        self.assertNotEqual(Sn1, Sn2)

    def test_check_has_next_stat(self):
        """test check has next state"""

        self.assertTrue( self.TC.has_next_state( (0,0), 'R', (0,1)) )
        self.assertFalse( self.TC.has_next_state( (0,0), 'R', (2,2)) )
        

    def test_remove_next_state(self):
        """test remove next state"""
        self.TC.set_transition( (0,0), 'R',
                                (0,3), reward_obj=Reward(const=1.0), 
                                action_prob=1.0, trans_prob=1.0)

        Sn1,p1 = self.TC.get_next_state_prob( (0,0), 'R', (0,1))
        Sn2,p2 = self.TC.get_next_state_prob( (0,0), 'R', (0,3))
        self.assertEqual(p1, 0.5)
        self.assertEqual(p1, p2)
        self.assertNotEqual(Sn1, Sn2)
        
        # now remove original transition
        self.TC.remove_next_state((0,0), 'R', (0,1))
        Sn1,p1 = self.TC.get_next_state_prob( (0,0), 'R', (0,1))
        Sn2,p2 = self.TC.get_next_state_prob( (0,0), 'R', (0,3))
        self.assertEqual(p1, None)
        self.assertNotEqual(p1, p2)
        self.assertNotEqual(Sn1, Sn2)
        self.assertEqual(p2, 1.0)

    def test_get_list_of_next_state_prob(self):
        """test get list of next state prob"""
        
        snpL = self.TC.get_list_of_all_next_state_prob( (0,0), 'R', incl_zero_prob=False)
        self.assertEqual( len(snpL), 1)
        
        snpL = self.TC.get_list_of_all_next_state_prob( (0,0), 'R', incl_zero_prob=True)
        self.assertEqual( len(snpL), 1)
        
        # add another transition
        self.TC.set_transition( (0,0), 'R',
                                (0,3), reward_obj=Reward(const=1.0), 
                                action_prob=1.0, trans_prob=1.0)
        
        snpL = self.TC.get_list_of_all_next_state_prob( (0,0), 'R', incl_zero_prob=False)
        self.assertEqual( len(snpL), 2)
        
        snpL = self.TC.get_list_of_all_next_state_prob( (0,0), 'R', incl_zero_prob=True)
        self.assertEqual( len(snpL), 2)
        
        # make one transition prob zero
        self.TC.initialize_sole_random( (0,0), 'R' )
        snpL = self.TC.get_list_of_all_next_state_prob( (0,0), 'R', incl_zero_prob=False)
        self.assertEqual( len(snpL), 1)
        
        snpL = self.TC.get_list_of_all_next_state_prob( (0,0), 'R', incl_zero_prob=True)
        self.assertEqual( len(snpL), 2)

    def test_get_list_of_next_state(self):
        """test get list of next state prob"""
        
        snL = self.TC.get_list_of_all_next_state( (0,0), 'R', incl_zero_prob=False)
        self.assertEqual( len(snL), 1)
        
        snL = self.TC.get_list_of_all_next_state( (0,0), 'R', incl_zero_prob=True)
        self.assertEqual( len(snL), 1)
        
        # add another transition
        self.TC.set_transition( (0,0), 'R',
                                (0,3), reward_obj=Reward(const=1.0), 
                                action_prob=1.0, trans_prob=1.0)
        
        snL = self.TC.get_list_of_all_next_state( (0,0), 'R', incl_zero_prob=False)
        self.assertEqual( len(snL), 2)
        
        snL = self.TC.get_list_of_all_next_state( (0,0), 'R', incl_zero_prob=True)
        self.assertEqual( len(snL), 2)
        
        # make one transition prob zero
        self.TC.initialize_sole_random( (0,0), 'R' )
        snL = self.TC.get_list_of_all_next_state( (0,0), 'R', incl_zero_prob=False)
        self.assertEqual( len(snL), 1)
        
        snL = self.TC.get_list_of_all_next_state( (0,0), 'R', incl_zero_prob=True)
        self.assertEqual( len(snL), 2)


    def test_iter_transitions(self):
        """test iter transitions"""
        
        # add another transition
        self.TC.set_transition( (0,0), 'R',
                                (0,3), reward_obj=Reward(const=1.0), 
                                action_prob=1.0, trans_prob=1.0)
        
        spL = []
        for (Sn, p) in self.TC.iter_next_state_prob( (0,0), 'R', incl_zero_prob=False):
            spL.append( p )

        self.assertEqual(spL, [0.5, 0.5])
        
        # make one transition prob zero
        self.TC.initialize_sole_random( (0,0), 'R' )
        
        spL = []
        for (Sn, p) in self.TC.iter_next_state_prob( (0,0), 'R', incl_zero_prob=False):
            spL.append( p )

        self.assertEqual(spL, [1.0])
        
        spL = []
        for (Sn, p) in self.TC.iter_next_state_prob( (0,0), 'R', incl_zero_prob=True):
            spL.append( p )

        self.assertEqual(sorted(spL), [0.0, 1.0])


    def test_get_random_transition(self):
        """test get random transition"""
        
        # add another transition
        self.TC.set_transition( (0,0), 'R',
                                (0,3), reward_obj=Reward(const=1.0), 
                                action_prob=1.0, trans_prob=1.0)
        snL = []
        for i in range(30):
            Sn = self.TC.get_prob_weighted_next_state( (0,0), 'R' )
            snL.append( Sn.hash )

        self.assertGreater(snL.count( (0,1) ), 5) # should be 15

    def test_terminal_set(self):
        """test terminal set"""
        
        term_set, action_set = self.TC.get_terminal_set_and_action_set()
        self.assertEqual(term_set, set( [(0,3), (1,3)] ))


if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

