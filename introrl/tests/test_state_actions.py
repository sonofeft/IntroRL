
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
from introrl.state_actions import StateActions

class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        s = State( (2,2) )
        self.sa = StateActions(s)
        for d in ['U','D','L','R']:
            a = Action(d)
            self.sa.set_action_prob( a, prob=len(self.sa))

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.sa )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""

        # See if the self.sa object exists
        self.assertIsInstance(self.sa, StateActions)

    def test_initialize(self):
        """test initialize"""
        AR,p = self.sa.get_action_prob_by_desc( 'R' )

        self.assertAlmostEqual(p, 0.5, delta=0.1)

    def test_random_action(self):
        """test random action"""
        aL = []
        for i in range(30):
            A = self.sa.get_prob_weighted_action()
            aL.append( A.desc )
        self.assertGreater(aL.count('R'), 5) # should be 15
        
        # check at 1 and 0 items in list
        self.sa.remove_action_by_desc( 'U' )
        self.sa.remove_action_by_desc( 'D' )
        self.sa.remove_action_by_desc( 'L' )
        
        A = self.sa.get_prob_weighted_action()
        self.assertEqual( A.desc, 'R' )

        self.sa.remove_action_by_desc( 'R' )
        A = self.sa.get_prob_weighted_action()
        self.assertEqual( A, None )


    def test_has_action(self):
        """test has action"""
        
        A,p = self.sa.get_action_prob_by_desc( 'L' )
        self.assertIsInstance(A, Action)
        self.assertEqual(A.desc, 'L')

        A,p = self.sa.get_action_prob_by_desc( 'xxx' )
        self.assertEqual(A, None)

        self.assertTrue( self.sa.get_action_prob_by_desc( 'U' ) )
        self.assertTrue( self.sa.get_action_prob_by_desc( 'D' ) )
        self.assertTrue( self.sa.get_action_prob_by_desc( 'L' ) )
        self.assertTrue( self.sa.get_action_prob_by_desc( 'R' ) )


        # test has_action_desc
        self.assertTrue( self.sa.has_action_desc( 'U' ) )
        self.assertTrue( self.sa.has_action_desc( 'D' ) )
        self.assertTrue( self.sa.has_action_desc( 'L' ) )
        self.assertTrue( self.sa.has_action_desc( 'R' ) )

        self.assertFalse( self.sa.has_action_desc( 'r' ) )

    def test_remove_action(self):
        """test remove action"""

        A,p = self.sa.get_action_prob_by_desc( 'L' )
        self.assertTrue( self.sa.has_action_desc( 'L' ) )
        self.sa.remove_action( A )
        self.assertFalse( self.sa.has_action_desc( 'L' ) )

        self.assertTrue( self.sa.has_action_desc( 'R' ) )
        self.sa.remove_action_by_desc( 'R' )
        self.assertFalse( self.sa.has_action_desc( 'R' ) )
        
        self.sa.remove_action_by_desc( 'xxx' )
        
        
    def test_get_all_actions_list(self):
        """test get all actions list"""
        apL = self.sa.get_list_of_all_action_prob( incl_zero_prob=False )
        self.assertEqual(len(apL), 3)

        apL = self.sa.get_list_of_all_action_prob( incl_zero_prob=True )
        self.assertEqual(len(apL), 4)

        self.assertEqual(len(apL[0]), 2)# should be an (A,prob) pair
        
        # now check just list of actions w/o probs
        aL = self.sa.get_list_of_all_actions( incl_zero_prob=False )
        self.assertEqual(len(aL), 3)
        
        aL = self.sa.get_list_of_all_actions( incl_zero_prob=True )
        self.assertEqual(len(aL), 4)

    def test_iterate_ap_pairs(self):
        """test iterate ap pairs"""
        apL = []
        for ap in self.sa.iter_action_prob( incl_zero_prob=False):        
            apL.append( ap )
        self.assertEqual(len(apL), 3)

        apL = []
        for ap in self.sa.iter_action_prob( incl_zero_prob=True):        
            apL.append( ap )
        self.assertEqual(len(apL), 4)
        self.assertEqual(len(ap), 2)

    def test_sole_action(self):
        """test sole action"""
        A,p1 = self.sa.get_action_prob_by_desc( 'L' )
        self.sa.set_sole_action( A )
        A,p2 = self.sa.get_action_prob_by_desc( 'L' )
        self.assertNotEqual(p1, p2)
        
        for d in ['U','D','L','R']:
            A,p = self.sa.get_action_prob_by_desc( d )
            if d == 'L':
                self.assertEqual(1.0, p)
            else:
                self.assertEqual(0.0, p)

        self.sa.set_sole_action_by_desc( 'R' )
        for d in ['U','D','L','R']:
            A,p = self.sa.get_action_prob_by_desc( d )
            if d == 'R':
                self.assertEqual(1.0, p)
            else:
                self.assertEqual(0.0, p)

        #for d in ['U','D','L','R']:

    def test_init_random_sole(self):
        """test init random sole"""

        self.sa.initialize_sole_random()
        num_zero = 0
        for d in ['U','D','L','R']:
            A,p = self.sa.get_action_prob_by_desc( d )
            if p==0.0:
                num_zero += 1
                
        self.assertEqual(num_zero, 3)

    def test_intialize_to_equiprobable(self):
        """test intialize_to_equiprobable"""
        
        self.sa.intialize_to_equiprobable()
        
        for d in ['U','D','L','R']:
            A,p = self.sa.get_action_prob_by_desc( d )
            self.assertEqual(p, 0.25)

    def test_all_zero_normalize(self):
        """test all zero normalize"""
        
        for d in ['U','D','L','R']:
            self.sa.set_action_prob_by_desc( d, prob=0.0)

        self.sa.normalize()
        
        apL = self.sa.get_list_of_all_action_prob( incl_zero_prob=True )
        
        print( [p for (A,p) in apL] )
        
        for d in ['U','D','L','R']:
            A,p = self.sa.get_action_prob_by_desc( d )
            self.assertEqual(p, 0.25)


if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

