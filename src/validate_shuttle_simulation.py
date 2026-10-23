"""Independent Monte Carlo evaluation of the computed shuttle policy."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parent.parent
    data=np.genfromtxt(root/'results/shuttle/values.csv',delimiter=',',names=True)
    policy=data['dispatch_policy'].astype(int)
    rng=np.random.default_rng(314159)
    starts=[0,12,13,15,200]
    n=40000; horizon=500; gamma=.95
    states=np.repeat(np.array(starts)[:,None],n,axis=1)
    total=np.zeros_like(states,dtype=float)
    # Explicit simulation of events, independent of the transition table in shuttle.py.
    discount=1.
    for _ in range(horizon):
        dispatch=policy[states]
        remaining=np.maximum(states-15*dispatch,0)
        total+=discount*(100*dispatch+2*remaining)
        states=np.minimum(200,remaining+rng.integers(1,6,size=states.shape))
        discount*=gamma
    rows=[]
    for s,samples in zip(starts,total):
        target=float(data['infinite_cost_pi'][s])
        mean=float(samples.mean()); se=float(samples.std(ddof=1)/np.sqrt(n))
        rows.append(dict(initial_state=s,episodes=n,simulated_mean=mean,dp_cost=target,
                         standard_error=se,ci95=[mean-1.96*se,mean+1.96*se],
                         simultaneous_ci95_bonferroni=[mean-2.576*se,mean+2.576*se],
                         standardized_discrepancy=(mean-target)/se))
    result=dict(seed=314159,horizon=horizon,discount=gamma,rows=rows,
                maximum_discounted_tail_bound=470*gamma**horizon/(1-gamma),
                interpretation='Intervals describe independent simulation uncertainty; they do not certify arbitrary model assumptions.')
    out=root/'results/shuttle/simulation_validation.json'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
