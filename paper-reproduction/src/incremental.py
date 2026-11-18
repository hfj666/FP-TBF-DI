"""Exact bounded-length itemset/coverage cache and equivalent faster selection.

This is an engineering comparator, not a replacement labeled as FP-growth.
All observed itemsets up to max_length are retained, including infrequent ones.
"""
from itertools import combinations
from math import ceil
import numpy as np
from .models import Rule
from .selection import CoveragePool


class PatternCoverageIndex:
    def __init__(self, max_length=3):
        if max_length < 1:
            raise ValueError('Positive maximum length required.')
        self.max_length=max_length
        self.n=0
        self.positive_mask=0
        self.stats={}

    def update(self,x,y):
        for row,label in zip(x,y,strict=True):
            bit=1<<self.n
            if label==1:
                self.positive_mask |= bit
            elif label!=-1:
                raise ValueError('Expected -1/+1 labels.')
            items=tuple((j,int(v)) for j,v in enumerate(row))
            for length in range(1,self.max_length+1):
                for pattern in combinations(items,length):
                    state=self.stats.setdefault(pattern,[0,0,0])
                    state[0]+=1
                    state[1]+=int(label==1)
                    state[2] |= bit
            self.n+=1
        return self

    def mine(self,min_support=.1):
        if not 0<min_support<=1:
            raise ValueError('Support must be in (0,1].')
        threshold=ceil(self.n*min_support)
        return {p:(s[0],s[1]) for p,s in self.stats.items() if s[0]>=threshold}

    def pool(self,patterns,confidence=.65,min_rows=5):
        pool=CachedCoveragePool.__new__(CachedCoveragePool)
        pool.patterns=patterns
        pool.pools={-1:[],1:[]}
        pool.own={1:self.positive_mask,-1:((1<<self.n)-1)^self.positive_mask}
        pool.class_n={label:mask.bit_count() for label,mask in pool.own.items()}
        if min(pool.class_n.values())==0:
            raise ValueError('Both classes required.')
        for items,(count,positive) in sorted(patterns.items()):
            purity=max(positive,count-positive)/count
            if count<min_rows or purity<=confidence:
                continue
            mask=self.stats[items][2]
            if mask.bit_count()!=count or (mask&self.positive_mask).bit_count()!=positive:
                raise AssertionError('Invalid cached coverage counters.')
            label=1 if positive>count/2 else -1
            rule=Rule(tuple(j for j,_ in items),np.array([v for _,v in items],dtype=float),label,purity,count)
            quality=(purity-.5)*np.sqrt(count)*np.sqrt(len(items))
            pool.pools[label].append((items,rule,mask,float(quality)))
        return pool


class CachedCoveragePool(CoveragePool):
    def select(self,alpha,max_rules=64):
        if not 0<=alpha<=1 or max_rules<2 or max_rules%2:
            raise ValueError('Invalid alpha or rule budget.')
        selected=[]
        for label in (-1,1):
            pool=self.pools[label]
            if not pool:
                continue
            scale=max(r[3] for r in pool)
            covered,remaining=0,list(range(len(pool)))
            overlap=[0.]*len(pool)
            for _ in range(min(max_rules//2,len(pool))):
                best,best_score=None,-np.inf
                for i in remaining:
                    _,rule,mask,quality=pool[i]
                    novelty=(mask&self.own[label]&~covered).bit_count()/self.class_n[label]
                    score=alpha*quality/scale+(1-alpha)*(novelty-overlap[i])
                    if score>best_score:
                        best,best_score=i,score
                _,rule,mask,_=pool[best]
                selected.append(rule)
                covered |= mask&self.own[label]
                remaining.remove(best)
                for i in remaining:
                    other=pool[i][2]
                    overlap[i]=max(overlap[i],(mask&other).bit_count()/(mask|other).bit_count())
        return selected
