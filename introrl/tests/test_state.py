
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

from introrl.state import State
from introrl.state_coll import StateColl
import introrl.state_coll

def mock_random_choice( vL ):
    return vL[ 1 % len(vL) ] # always return second item.

class MyTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.s = State( 'X-OX-OX--' )

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        del( self.s )

    def test_should_always_pass_cleanly(self):
        """Should always pass cleanly."""
        pass

    def test_myclass_existence(self):
        """Check that myclass exists"""
        result = self.s

        # See if the self.s object exists
        self.assertTrue(result)

    def test_check_hash_property(self):
        """check hash property"""

        self.assertEqual(self.s.hash, 'X-OX-OX--')

    def test_check_tuple_of_values(self):
        """check tuple of values"""
        s = State( (1, 'Z') )
        self.assertEqual(s.hash, (1, 'Z'))
        print( s )

    def test_iteration(self):
        """test iteration"""
        sc = StateColl( name='Testing' )
        for d in ['----X---O', ('a',3), 47]:
            sc.add_state( d )
        self.assertEqual( len(sc), 3 )
        
        self.assertTrue( sc.has_state_hash( 47 ) )
        self.assertFalse( sc.has_state_hash( 48 ) )

        outL = [S.hash for S in sc.iter_states()]
        self.assertEqual( len(outL), 3 )
        for v in outL:
            self.assertIn(v, ['----X---O', ('a',3), 47]) # S.id should be in order
            
        outL = [s_hash for s_hash in sc.iter_state_hash()]
        self.assertEqual( len(outL), 3 )
        for v in outL:
            self.assertIn(v, ['----X---O', ('a',3), 47]) # S.id should be in order

    def test_random_choice(self):
        """test random choice"""
        sc = StateColl( name='Testing' )
        for d in ['----X---O', ('a',3), 47]:
            sc.add_state( d )

        introrl.state_coll.random.choice = lambda vL: mock_random_choice( vL )

        S = sc.get_random_state()
        self.assertIn(S.hash, ['----X---O', ('a',3), 47])

if __name__ == '__main__':
    # Can test just this file from command prompt
    #  or it can be part of test discovery from nose, unittest, pytest, etc.
    unittest.main()

