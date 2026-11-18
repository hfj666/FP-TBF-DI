"""Author-draft zero-order eClass recurrence and local sparse Sugeno updates.

These are transparent research implementations, not author-certified replicas.
The eClass source PDF/equation mapping and under-specified choices are documented
in online_reference_notes.md. Neither class stores raw historical observations.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp,expit
from .models import Rule


@dataclass
class Prototype:
    center: np.ndarray
    potential: float
    variance: float
    support: int = 1


@dataclass
class ClassDensity:
    n: int
    vector_sum: np.ndarray
    square_sum: float
    prototypes: list


class EClass0:
    def __init__(self,minimum,span,sigma_floor=.1):
        self.minimum=np.array(minimum,copy=True)
        self.span=np.array(span,copy=True)
        self.sigma_floor=sigma_floor
        self.classes={}
        self.samples_updated=0
        self.rules_added=0
        self.rules_removed=0
        self.potential_evaluations=0

    def new_potential(self,z,label):
        state=self.classes[label]
        # Eq(7)-(10), with class-local count, before consuming current sample.
        distance_sum=state.n*float(z@z)+state.square_sum-2*float(z@state.vector_sum)
        return state.n/(state.n+max(0.,distance_sum))

    @staticmethod
    def cosine_distance(a,b):
        norm=np.linalg.norm(a)*np.linalg.norm(b)
        if norm==0:
            return 0. if np.array_equal(a,b) else 1.
        return float(np.clip(1-(a@b)/norm,0.,2.))

    def partial_fit(self,x,y):
        for raw,label in zip(x,y,strict=True):
            label=int(label);z=(raw-self.minimum)/self.span
            self.samples_updated+=1
            if label not in self.classes:
                self.classes[label]=ClassDensity(1,z.copy(),float(z@z),[Prototype(z.copy(),1.,1.)])
                self.rules_added+=1
                continue
            state=self.classes[label];m=state.n
            potential=self.new_potential(z,label)
            self.potential_evaluations+=len(state.prototypes)+1
            # Eq(11). Old prototype excludes its own zero-distance observation.
            for prototype in state.prototypes:
                distance=float(np.sum((z-prototype.center)**2))
                p=prototype.potential
                prototype.potential=m*p/((m-1)+p*(1+distance))
            if potential>max(p.potential for p in state.prototypes)+1e-14:
                spread=float(np.mean([np.sqrt(p.variance) for p in state.prototypes]))
                new=Prototype(z.copy(),potential,max(spread,self.sigma_floor)**2)
                retained=[]
                # Eq(13), existing-rule Gaussian memberships, all coordinates.
                for p in state.prototypes:
                    if np.all(np.abs(z-p.center)<np.sqrt(2*max(p.variance,self.sigma_floor**2))):
                        self.rules_removed+=1
                    else:
                        retained.append(p)
                state.prototypes=retained+[new]
                self.rules_added+=1
            else:
                nearest=min(state.prototypes,key=lambda p:float(np.sum((z-p.center)**2)))
                nearest.support+=1
                distance=self.cosine_distance(z,nearest.center)
                nearest.variance=max(self.sigma_floor**2,nearest.variance+
                                     (distance**2-nearest.variance)/nearest.support)
            state.n+=1;state.vector_sum+=z;state.square_sum+=float(z@z)
        return self

    @property
    def rule_count(self):return sum(len(s.prototypes) for s in self.classes.values())

    def decision_function(self,x):
        z=(x-self.minimum)/self.span
        logs={}
        for label,state in self.classes.items():
            logfire=np.column_stack([-.5*np.sum((z-p.center)**2,axis=1)/max(p.variance,self.sigma_floor**2)
                                     for p in state.prototypes])
            logs[label]=logsumexp(logfire,axis=1)
        # Sum of normalized one-hot consequent activations, Eq(1b,c),(2).
        # Log-odds has the identical argmax and avoids Gaussian underflow.
        return logs.get(1,np.full(len(x),-1e12))-logs.get(-1,np.full(len(x),-1e12))


class LocalSugeno:
    def __init__(self,initial_model,learning_rate=.05,active_k=8):
        self.minimum=initial_model.minimum.copy();self.span=initial_model.span.copy()
        self.rules=[Rule(r.columns,r.values.copy(),r.label,r.confidence,r.count) for r in initial_model.rules]
        self.theta=initial_model.theta.copy()
        self.learning_rate=learning_rate;self.active_k=active_k
        self.samples_updated=0;self.parameter_rule_updates=0

    def distances(self,z):
        return np.array([np.mean((z[list(r.columns)]-r.values)**2) for r in self.rules])

    def replace_rules(self,raw_rules):
        previous={(r.columns,tuple(np.round(r.values*self.span[list(r.columns)]+self.minimum[list(r.columns)],9))):
                  (i,r.label) for i,r in enumerate(self.rules)}
        old=self.theta;old_k=len(self.rules);new_k=len(raw_rules)
        new=np.r_[np.full(new_k,np.log(.25)),[2*r.label for r in raw_rules]]
        kept=0
        for i,r in enumerate(raw_rules):
            match=previous.get((r.columns,tuple(np.round(r.values,9))))
            if match is not None and match[1]==r.label:
                new[i]=old[match[0]];new[new_k+i]=old[old_k+match[0]];kept+=1
        self.rules=[Rule(r.columns,(r.values-self.minimum[list(r.columns)])/self.span[list(r.columns)],
                         r.label,r.confidence,r.count) for r in raw_rules]
        self.theta=new
        return dict(retained=kept,new=new_k-kept,removed=old_k-kept,metadata_refreshed=new_k)

    def step(self,raw,label):
        z=(raw-self.minimum)/self.span;k=len(self.rules)
        d2=self.distances(z);a=d2*np.exp(-2*self.theta[:k]);c=self.theta[k:].copy()
        logfire=-.5*a+np.log([r.confidence for r in self.rules])
        weight=np.exp(logfire-logsumexp(logfire));score=float(weight@c)
        error=float(expit(score)-(label==1))
        active=np.argsort(-weight,kind='stable')[:min(self.active_k,k)]
        grad_c=error*weight+.002*c/k
        grad_w=error*weight*(c-score)*a
        self.theta[active]=np.clip(self.theta[active]-self.learning_rate*grad_w[active],np.log(.03),np.log(2.))
        self.theta[k+active]=np.clip(c[active]-self.learning_rate*grad_c[active],-10.,10.)
        self.samples_updated+=1;self.parameter_rule_updates+=len(active)
        return active

    def partial_fit(self,x,y):
        for row,label in zip(x,y,strict=True):self.step(row,int(label))
        return self

    def decision_function(self,x):
        # Same vectorized evaluation as the full model, so prediction timing does
        # not compare a Python row loop with a vectorized baseline.
        k=len(self.rules);z=(x-self.minimum)/self.span
        d2=np.column_stack([np.mean((z[:,list(r.columns)]-r.values)**2,axis=1) for r in self.rules])
        logfire=-.5*d2*np.exp(-2*self.theta[:k])+np.log([r.confidence for r in self.rules])
        weights=np.exp(logfire-logsumexp(logfire,axis=1)[:,None])
        return weights@self.theta[k:]
