"""
Need to overcome python 3 errors for sorting mixed types
"""

class NaturalOrStrKey:
    """
    If there is a natural key that works for sorting, do it.
    Otherwise, use str.
    
    NOTE: python 2 and 3 will result in DIFFERENT sorted lists.
    """

    __slots__ = ("value", "strval")

    def __init__(self, value):
        self.value   = value
        self.strval =  str(value)

    def __lt__(self, other):
        try:
            return self.value < other.value
        except TypeError:
            #print('self.strval="%s", other.strval="%s" '%(self.strval, other.strval) )
            return self.strval < other.strval


if __name__ == "__main__": # pragma: no cover
    
    myL = [ 9, ('B',3), 'seven', (2,3), ('a',2), (2,2), 1, ('b',3), (3, 'b'), 2.3, 11.1 ]
    
    print( sorted( myL, key=NaturalOrStrKey) )
    
    print("""         --> NOTE DIFFERENT RESULTS FOR PYTHON 2 AND 3 <--
    python 2.7
[1, 2.3, 9, 11.1, 'seven', (2, 2), (2, 3), (3, 'b'), ('B', 3), ('a', 2), ('b', 3)]

    python 3.7
[('B', 3), ('a', 2), ('b', 3), (2, 2), (2, 3), (3, 'b'), 1, 2.3, 9, 11.1, 'seven']
    """)
