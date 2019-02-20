
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
from introrl.reward import Reward

reward_probL = [(0.0,1), (1.0,1), (2.0,2)]

def my_gauss():
    return random.gauss(3.0, 0.5)


class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.Rc = Reward( const=1.1 )
        self.Rt = Reward( reward_probL=reward_probL )
        self.Rf = Reward( reward_dist_func=my_gauss )

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.Rc )
        del( self.Rt )
        del( self.Rf )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""
        
        # See if the self.R object exists
        self.assertIsInstance(self.Rc, Reward, msg=None)
        self.assertIsInstance(self.Rt, Reward, msg=None)
        self.assertIsInstance(self.Rf, Reward, msg=None)

    def test_str(self):
        """test str"""
        sc = str( self.Rc )
        st = str( self.Rt )
        sf = str( self.Rf )
        self.assertEqual(sc, '<Reward-Constant = 1.1>')
        self.assertEqual(st, '<Reward-Tabular = [(0.0, 1), (1.0, 1), (2.0, 2)]>')
        self.assertEqual(sf, '<Reward-Function = my_gauss>')

    def test_call(self):
        """test call"""
        vc = self.Rc()
        self.assertEqual(vc, 1.1)

        vt = self.Rt()
        self.assertIn(vt, [0.0, 1.0, 2.0])

        vf = self.Rf()
        self.assertAlmostEqual(vf, 3.0, delta=10.0)


if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

