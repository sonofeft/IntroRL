import heapq

class SweepPriorityQueue:
    """
    A priority queue used for prioritized sweeping algorithm 
    from page 169 of Sutton & Barto
    
    State-Action pairs are added along with the NEGATIVE of abs( delta )
    where delta is the expected change in Q(s,a)
    """
    def __init__(self):
        self.qlist = []
        self.queue_entryD = {}
        self.nth_addition = 0

    def add_sa_pair(self, sa_pair, neg_priority=0):
        if sa_pair in self.queue_entryD:
            old_neg_priority = self.queue_entryD[ sa_pair ][0]
            if old_neg_priority < neg_priority:
                return # old entry is higher priority, so leave it.
                
            # new value is higher priority, so mark old one for deletion
            self.mark_sa_pair_for_deletion(sa_pair)
        entry = [neg_priority, self.nth_addition, sa_pair]
        self.nth_addition += 1
        self.queue_entryD[sa_pair] = entry
        heapq.heappush(self.qlist, entry)

    def mark_sa_pair_for_deletion(self, sa_pair):
        entry = self.queue_entryD.pop(sa_pair)
        entry[-1] = None

    def pop_sa_pair(self):
        while self.qlist:
            neg_priority, _, sa_pair = heapq.heappop(self.qlist)
            if sa_pair is not None:
                del self.queue_entryD[sa_pair]
                return sa_pair, neg_priority
        return (None,None), None

    def empty(self):
        return not self.queue_entryD
        
    def summ_print(self):
        print('========= Current Priority Queue Contents ===========')
        for v in self.qlist:
            print( v )

if __name__ == "__main__": # pragma: no cover
    
    PQ = SweepPriorityQueue()
    
    PQ.add_sa_pair( ((1,1),'U'), -1 )
    PQ.add_sa_pair( ((1,2),'L'), -2 )
    PQ.add_sa_pair( ((2,2),'R'), -3 )
    PQ.add_sa_pair( ((3,3),'U'), -1 )
    PQ.add_sa_pair( ((2,2),'U'), -1 )
    PQ.add_sa_pair( ((1,1),'U'), 0 )

    for v in PQ.qlist:
        print( v )
    print('='*66)
    PQ.add_sa_pair( ((1,1),'U'), -2 )
    for v in PQ.qlist:
        print( v )
    print('='*66)
    while not PQ.empty():
        print( PQ.pop_sa_pair() )
    