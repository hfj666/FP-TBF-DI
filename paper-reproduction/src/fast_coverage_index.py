"""Equivalent bounded itemset updates with less Python allocation/conversion.

Retains all observed low-frequency itemsets and the same exact history bitmaps.
No bounded-memory or new-mining-algorithm claim.
"""
from itertools import combinations
from .incremental import PatternCoverageIndex


class FastPatternCoverageIndex(PatternCoverageIndex):
    def update(self,x,y):
        lookup=self.stats.get
        for row,label in zip(x,y,strict=True):
            bit=1<<self.n
            if label==1:
                positive=1;self.positive_mask|=bit
            elif label==-1:positive=0
            else:raise ValueError('Expected -1/+1 labels.')
            items=tuple((j,int(v)) for j,v in enumerate(row))
            for length in range(1,self.max_length+1):
                for pattern in combinations(items,length):
                    state=lookup(pattern)
                    if state is None:self.stats[pattern]=[1,positive,bit]
                    else:
                        state[0]+=1;state[1]+=positive;state[2]|=bit
            self.n+=1
        return self
