
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

from introrl.action import Action
from introrl.state import State
from introrl.transition import Transition
from introrl.reward import Reward
import random 

class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        s = State( (2,2) )
        a = Action( 'U' )
        self.T = Transition(s,a)
    
        rc = Reward(const=1.1)
        reward_probL = [(0.0,1), (1.0,1), (2.0,2)]
        rt = Reward(reward_probL=reward_probL)
        
        def my_gauss():
            return random.gauss(3.0, 0.5)
        rf = Reward(reward_dist_func=my_gauss)
        
        
        self.T.set_transition( State( (2,3) ), reward_obj=rc, prob=0.8)
        self.T.set_transition( State( (1,2) ), reward_obj=rt, prob=0.1)
        self.T.set_transition( State( (3,2) ), reward_obj=rf, prob=0.1)
        self.T.set_transition( State( (0,0) ), reward_obj=rf, prob=0.0)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.T )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""
        
        # See if the self.T object exists
        self.assertIsInstance(self.T, Transition, msg=None)
        
        self.assertEqual( len(self.T), 4)
        
        # check out Reward object retrieval
        R = self.T.get_reward_obj( (2,3) )
        self.assertEqual( R(), 1.1)

    def test_random_next_state(self):
        """test random next state"""
        snL = []
        for i in range(30):
            Sn = self.T.get_prob_weighted_next_state()
            snL.append( Sn.hash )
        self.assertGreater(snL.count((2,3)), 5) # should be 30*0.8
        
        # check for 1 and 0 Sn
        self.T.remove_next_state_by_hash( (0,0) )
        self.T.remove_next_state_by_hash( (2,3) )
        self.T.remove_next_state_by_hash( (1,2) )
        
        Sn = self.T.get_prob_weighted_next_state()
        self.assertEqual( Sn.hash, (3,2))

        self.T.remove_next_state_by_hash( (3,2) )
        Sn = self.T.get_prob_weighted_next_state()
        self.assertEqual( Sn, None)


    def test_has_next_state_by_hash(self):
        """test has next state"""
        
        Sn,p = self.T.get_next_state_prob_by_hash( (1,2) )
        self.assertIsInstance(Sn, State)
        self.assertEqual(Sn.hash, (1,2))
        
        # test has_next_state_by_hash
        self.assertTrue( self.T.has_next_state_by_hash( (2,3) ) )
        self.assertTrue( self.T.has_next_state_by_hash( (1,2) ) )
        self.assertTrue( self.T.has_next_state_by_hash( (3,2) ) )
        self.assertTrue( self.T.has_next_state_by_hash( (0,0) ) )

        self.assertFalse( self.T.has_next_state_by_hash( 'r' ) )

        self.assertTrue( self.T.has_next_state( Sn ) )

    def test_remove_next_state(self):
        """test remove next state"""
        
        snext_hash = (2,3)
        Sn,p = self.T.get_next_state_prob_by_hash( snext_hash )
        self.assertTrue( self.T.has_next_state_by_hash( snext_hash ) )
        self.T.remove_next_state_by_hash( snext_hash )
        self.assertFalse( self.T.has_next_state_by_hash( snext_hash ) )

        snext_hash = (1,2)
        self.assertTrue( self.T.has_next_state_by_hash( snext_hash ) )
        self.T.remove_next_state_by_hash( snext_hash )
        self.assertFalse( self.T.has_next_state_by_hash( snext_hash ) )
        
        # remove a state already removed
        self.T.remove_next_state( Sn )
        
    def test_get_all_next_state_list(self):
        """test get all next state list"""
        snpL = self.T.get_list_of_all_next_state_prob( incl_zero_prob=False )
        self.assertEqual(len(snpL), 3)

        snpL = self.T.get_list_of_all_next_state_prob( incl_zero_prob=True )
        self.assertEqual(len(snpL), 4)

        self.assertEqual(len(snpL[0]), 2)# should be an (Sn,prob) pair
        
        # get list w/o prob included
        snL = self.T.get_list_of_all_next_state( incl_zero_prob=False )
        self.assertEqual(len(snL), 3)
        snL = self.T.get_list_of_all_next_state( incl_zero_prob=True )
        self.assertEqual(len(snL), 4)

    def test_iterate_snp_pairs(self):
        """test iterate (Sn,p) pairs"""
        snpL = []
        for snp in self.T.iter_next_state_prob( incl_zero_prob=False):        
            snpL.append( snp )
        self.assertEqual(len(snpL), 3)

        snpL = []
        for snp in self.T.iter_next_state_prob( incl_zero_prob=True):        
            snpL.append( snp )
        self.assertEqual(len(snpL), 4)
        self.assertEqual(len(snp), 2)

    def test_sole_next_state(self):
        """test sole next_state"""
        
        snext_hash = (0,0)
        Sn,p1 = self.T.get_next_state_prob_by_hash( snext_hash )
        self.T.set_sole_transition_by_desc( snext_hash )
        
        Sn,p2 = self.T.get_next_state_prob_by_hash( snext_hash )
        self.assertNotEqual(p1, p2)
        
        for d in [(0,0), (3,2), (1,2), (2,3)]:
            Sn,p = self.T.get_next_state_prob_by_hash( d )
            if d == (0,0):
                self.assertEqual(1.0, p)
            else:
                self.assertEqual(0.0, p)

        self.T.set_sole_transition_by_desc( (1,2) )
        for d in [(0,0), (3,2), (1,2), (2,3)]:
            A,p = A,p2 = self.T.get_next_state_prob_by_hash( d )
            if d == (1,2):
                self.assertEqual(1.0, p)
            else:
                self.assertEqual(0.0, p)


    def test_init_random_sole(self):
        """test init random sole"""

        self.T.initialize_sole_random()
        num_zero = 0
        for d in [(0,0), (3,2), (1,2), (2,3)]:
            Sn,p = self.T.get_next_state_prob_by_hash( d )
            if p==0.0:
                num_zero += 1
                
        self.assertEqual(num_zero, 3)


    def test_intialize_to_equiprobable(self):
        """test intialize_to_equiprobable"""
        
        self.T.intialize_to_equiprobable()
        num_zero = 0
        for d in [(0,0), (3,2), (1,2), (2,3)]:
            Sn,p = self.T.get_next_state_prob_by_hash( d )
            self.assertEqual(p, 0.25)

    def test_normalize_all_zero_probs(self):
        """test normalize all zero probs"""
        self.T.set_transition_prob_by_snext_hash( (2,3), prob=0.0)
        self.T.set_transition_prob_by_snext_hash( (1,2), prob=0.0)
        self.T.set_transition_prob_by_snext_hash( (3,2), prob=0.0)
        self.T.set_transition_prob_by_snext_hash( (0,0), prob=0.0)
        
        #self.T.normalize()

        for snext_hash in [(0,0), (3,2), (1,2), (2,3)]:
            Sn,p = self.T.get_next_state_prob_by_hash( snext_hash )
            print('Got ',Sn.hash,' p= ',p)
            self.assertEqual(p, 0.25)
            

    def test_snhash_prob_reward_iterate(self):
        """test snhash prob reward iterate"""
        outL = []
        for sn_hash, t_prob, reward in self.T.iter_sn_hash_prob_reward():
            outL.append( t_prob )
        
        self.assertEqual(len(outL), 4)
        

if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

