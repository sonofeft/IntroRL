import itertools

class CircularList( object ):
    """
    CircularList datatype.
    """

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return itertools.cycle(self._items)

    def _from_integer_index(self, idx):
        if not isinstance(idx, int):
            raise TypeError("CircularList indices must be integers, not {}".format(type(idx)))

        if not len(self._items):
            raise IndexError("Indexing empty CircularList")

        return idx % len(self._items)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            step = 1 if idx.step is None else idx.step
            return [self[i] for i in xrange(idx.start, idx.stop, step)]

        return self._items[self._from_integer_index(idx)]

    def __setitem__(self, idx, value):
        self._items[self._from_integer_index(idx)] = value

    def __delitem__(self, idx):
        del self._items[self._from_integer_index(idx)]

    def __repr__(self):
        return "CircularList({})".format(self._items)
        
        
if __name__ == "__main__":  # pragma: no cover
    
    CL = CircularList([0,1,2])
    for i in range(5):
        print(i, CL[i])
    print('------------')
    CL[4] = 4
    for i in range(5):
        print(i, CL[i])
