
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

from introrl.state_actions_coll import StateActionsColl

class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.SAC = StateActionsColl()
        
        s_hash = (2,2)
        self.SAC.add_state_action( s_hash )
        prob = 0.0
        for a_desc in ['U','D','L','R']:
            self.SAC.set_action_prob( s_hash, a_desc, prob=prob)
            prob += 1.0
        
        s_hash = 14
        self.SAC.add_state_action( s_hash )
        for a_desc in ['Hit','Stay']:
            self.SAC.set_action_prob( s_hash, a_desc, prob=0.5)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.SAC )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""
        
        # See if the self.SAC object exists
        self.assertIsInstance(self.SAC, StateActionsColl, msg=None)
        
        self.assertEqual( len(self.SAC), 2 )

    def test_initialize(self):
        """test initialize"""
        p = self.SAC.get_action_prob((2,2), 'R')

        self.assertAlmostEqual(p, 0.5, delta=0.1)

    def test_random_action(self):
        """test random action"""
        aL = []
        for i in range(30):
            A = self.SAC.get_prob_weighted_action( (2,2) )
            aL.append( A.desc )
        self.assertGreater(aL.count('R'), 5) # should be 15


    def test_has_action(self):
        """test has action"""
        
        # test 1st StateAction 
        s_hash = (2,2)
        p = self.SAC.get_action_prob( s_hash, 'R' )
        self.assertEqual(p, 0.5)

        # test has_action
        self.assertTrue( self.SAC.has_action( s_hash, 'U' ) )
        self.assertTrue( self.SAC.has_action( s_hash, 'D' ) )
        self.assertTrue( self.SAC.has_action( s_hash, 'L' ) )
        self.assertTrue( self.SAC.has_action( s_hash, 'R' ) )

        self.assertFalse( self.SAC.has_action( s_hash, 'r' ) )
        
        # test 2nd StateAction
        
        s_hash = 14
        p = self.SAC.get_action_prob( s_hash, 'Hit' )
        self.assertEqual(p, 0.5)

        # test has_action
        self.assertTrue( self.SAC.has_action( s_hash, 'Hit' ) )
        self.assertTrue( self.SAC.has_action( s_hash, 'Stay' ) )

        self.assertFalse( self.SAC.has_action( s_hash, 'r' ) )

    def test_remove_action(self):
        """test remove action"""
        
        s_hash = (2,2)
        p = self.SAC.get_action_prob( s_hash, 'L' )
        self.assertTrue( self.SAC.has_action( s_hash, 'L' ) )
        self.SAC.remove_action( s_hash, 'L' )
        self.assertFalse( self.SAC.has_action( s_hash, 'L' ) )

        self.assertTrue( self.SAC.has_action( s_hash, 'R' ) )
        self.SAC.remove_action( s_hash, 'R' )
        self.assertFalse( self.SAC.has_action( s_hash, 'R' ) )
        
    def test_get_all_actions_list(self):
        """test get all actions list"""
        
        s_hash = (2,2)
        apL = self.SAC.get_list_of_all_action_desc_prob( s_hash, incl_zero_prob=False )
        self.assertEqual(len(apL), 3)

        apL = self.SAC.get_list_of_all_action_desc_prob( s_hash, incl_zero_prob=True )
        self.assertEqual(len(apL), 4)

        self.assertEqual(len(apL[0]), 2)# should be an (A,prob) pair
        
        # get a list of just the action descriptions
        adescL = self.SAC.get_list_of_all_action_desc( s_hash, incl_zero_prob=False )
        self.assertEqual(len(adescL), 3)
        adescL = self.SAC.get_list_of_all_action_desc( s_hash, incl_zero_prob=True )
        self.assertEqual(len(adescL), 4)
        
    def test_iterate_ap_pairs(self):
        """test iterate ap pairs"""
        
        s_hash = (2,2)
        apL = []
        for ap in self.SAC.iter_action_desc_prob( s_hash, incl_zero_prob=False):        
            apL.append( ap )
        self.assertEqual(len(apL), 3)

        apL = []
        for ap in self.SAC.iter_action_desc_prob( s_hash, incl_zero_prob=True):        
            apL.append( ap )
        self.assertEqual(len(apL), 4)
        self.assertEqual(len(ap), 2)
        
        # iterate shash, adesc, prob

        apL = []
        for ap in self.SAC.iter_shash_adesc_prob( incl_zero_prob=True):        
            apL.append( ap )
        self.assertEqual(len(apL), 6)
        self.assertEqual(len(ap), 3)

    def test_sole_action(self):
        """test sole action"""
        
        s_hash = (2,2)
        p1 = self.SAC.get_action_prob( s_hash, 'L' )
        self.SAC.set_sole_action( s_hash, 'L' )
        p2 = self.SAC.get_action_prob( s_hash, 'L' )
        self.assertNotEqual(p1, p2)
        
        for d in ['U','D','L','R']:
            p = self.SAC.get_action_prob( s_hash, d )
            if d == 'L':
                self.assertEqual(1.0, p)
            else:
                self.assertEqual(0.0, p)

        self.SAC.set_sole_action( s_hash, 'R' )
        for d in ['U','D','L','R']:
            p = self.SAC.get_action_prob( s_hash, d )
            if d == 'R':
                self.assertEqual(1.0, p)
            else:
                self.assertEqual(0.0, p)

    def test_init_random_sole(self):
        """test init random sole"""
        
        s_hash = (2,2)
        self.SAC.initialize_sole_random( s_hash )
        num_zero = 0
        for d in ['U','D','L','R']:
            p = self.SAC.get_action_prob( s_hash, d )
            if p==0.0:
                num_zero += 1
                
        self.assertEqual(num_zero, 3)

    def test_intialize_to_equiprobable(self):
        """test intialize_to_equiprobable"""
        
        s_hash = (2,2)
        self.SAC.intialize_to_equiprobable( s_hash )
        num_zero = 0
        for d in ['U','D','L','R']:
            p = self.SAC.get_action_prob( s_hash, d )
            self.assertEqual(p, 0.25)
        

if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

