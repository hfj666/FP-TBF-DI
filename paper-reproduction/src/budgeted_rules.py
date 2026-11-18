"""Candidate method: bounded rule exchange with an exact utility upper bound.

This defines a NEW, explicit set objective. It is not equivalent to the old MMR
selector, and is not yet claimed novel. Monotone coverage/submodularity, bounded
recourse and branch-and-bound are existing ideas.
"""
import numpy as np


def identity(entry):return (entry[0],entry[1].label)


def class_data(pool,label,budget,alpha):
    rows=pool.pools[label];k=min(budget,len(rows))
    scale=max((r[3] for r in rows),default=1.)
    # Half of the total objective is assigned to each diagnostic class.
    modular=np.array([alpha*r[3]/(2*k*scale) for r in rows]) if k else np.empty(0)
    coverage=(1-alpha)/(2*pool.class_n[label])
    masks=[r[2]&pool.own[label] for r in rows]
    return dict(rows=rows,k=k,modular=modular,coverage=coverage,masks=masks)


def utility(data,selected):
    value=0.
    for label in (-1,1):
        d=data[label];covered=0
        for i in selected[label]:covered|=d['masks'][i];value+=float(d['modular'][i])
        value+=d['coverage']*covered.bit_count()
    return value


def fill_greedy(d,selected):
    """Vectorized exact marginal-gain greedy, also used as the global comparator."""
    if not d['k']:return selected
    n=len(d['rows']);words=max(1,(max(d['masks']).bit_length()+63)//64)
    packed=np.frombuffer(b''.join(m.to_bytes(words*8,'little') for m in d['masks']),dtype='<u8').reshape(n,words)
    covered=np.zeros(words,dtype=np.uint64);eligible=np.ones(n,dtype=bool)
    for i in selected:covered|=packed[i];eligible[i]=False
    while len(selected)<d['k']:
        added=np.bitwise_count(packed&~covered).sum(axis=1,dtype=np.int64)
        gains=d['modular']+d['coverage']*added;gains[~eligible]=-np.inf
        best=int(np.argmax(gains));selected.append(best);covered|=packed[best];eligible[best]=False
    return selected


def swap_context(d,selected):
    masks=[d['masks'][i] for i in selected];prefix=[0]
    for mask in masks:prefix.append(prefix[-1]|mask)
    suffix=[0]*(len(masks)+1)
    for j in range(len(masks)-1,-1,-1):suffix[j]=suffix[j+1]|masks[j]
    exclusive=[mask&~(prefix[j]|suffix[j+1]) for j,mask in enumerate(masks)]
    return prefix[-1],exclusive


def best_swap(data,selected,bounded=True):
    """Upper bound = modular gain against min quality + newly covered cases.

    Exact gain subtracts the removed rule's exclusive coverage not recovered by
    the incoming rule. That subtraction is nonnegative, hence this is an upper
    bound. Numerical guard1e-12 only prevents last-bit bound violations.
    """
    candidates=[];contexts={};exact_pairs=0
    for label in (-1,1):
        d=data[label];chosen=selected[label]
        if not chosen or len(chosen)==len(d['rows']):continue
        covered,exclusive=swap_context(d,chosen);contexts[label]=(covered,exclusive)
        chosen_set=set(chosen);minimum=min(float(d['modular'][i]) for i in chosen)
        for incoming in range(len(d['rows'])):
            if incoming in chosen_set:continue
            added=(d['masks'][incoming]&~covered).bit_count()
            upper=float(d['modular'][incoming])-minimum+d['coverage']*added
            candidates.append((label,incoming,added,upper))
    if bounded:candidates.sort(key=lambda r:(-r[3],r[0],r[1]))
    best_gain=0.;best_key=None;best=None;visited=0
    for label,incoming,added,upper in candidates:
        if bounded and upper+1e-12<best_gain:break
        visited+=1;d=data[label];_,exclusive=contexts[label]
        for position,outgoing in enumerate(selected[label]):
            lost=(exclusive[position]&~d['masks'][incoming]).bit_count()
            gain=float(d['modular'][incoming]-d['modular'][outgoing])+d['coverage']*(added-lost)
            assert gain<=upper+1e-12
            exact_pairs+=1;key=(label,incoming,position)
            if gain>best_gain or (gain==best_gain and best_key is not None and key<best_key):
                best_gain=gain;best_key=key;best=(label,incoming,position)
    return best,best_gain,dict(exact_pairs=exact_pairs,candidates=len(candidates),
                               candidates_skipped=len(candidates)-visited)


def select_budgeted(pool,previous=None,alpha=.5,max_rules=64,max_swaps=4,bounded=True,global_rebuild=False):
    if not 0<=alpha<=1 or max_rules<2 or max_rules%2 or max_swaps<0:
        raise ValueError('invalid objective/budget')
    data={label:class_data(pool,label,max_rules//2,alpha) for label in (-1,1)}
    previous=list(previous or []);selected={-1:[],1:[]};forced_removed=0;forced_added=0
    if previous and not global_rebuild:
        lookup={identity(row):(label,i) for label,d in data.items() for i,row in enumerate(d['rows'])}
        for key in previous:
            found=lookup.get(key)
            if found is None:forced_removed+=1
            else:selected[found[0]].append(found[1])
    for label in (-1,1):
        before=len(selected[label]);fill_greedy(data[label],selected[label])
        forced_added+=len(selected[label])-before
    starting=utility(data,selected);swaps=exact_pairs=skipped=0;stopped_no_gain=False
    if previous and not global_rebuild:
        for _ in range(max_swaps):
            swap,gain,counters=best_swap(data,selected,bounded)
            exact_pairs+=counters['exact_pairs'];skipped+=counters['candidates_skipped']
            if swap is None or gain<=1e-12:stopped_no_gain=True;break
            label,incoming,position=swap;before=utility(data,selected)
            selected[label][position]=incoming;after=utility(data,selected)
            assert abs((after-before)-gain)<1e-10 and after>before
            swaps+=1
    entries=[data[label]['rows'][i] for label in (-1,1) for i in selected[label]]
    return [e[1] for e in entries],[identity(e) for e in entries],dict(
        objective=utility(data,selected),objective_before_swaps=starting,
        discretionary_swaps=swaps,forced_removed=forced_removed,forced_added=forced_added,
        exact_pairs=exact_pairs,candidates_skipped=skipped,stopped_no_gain=stopped_no_gain)
