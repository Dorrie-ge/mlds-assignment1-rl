"""Check a real network derivative; measure gradient variance at a frozen policy.

Example:
 python src/validate_policy_gradient.py --game cartpole \
   --checkpoint results/cartpole_value/checkpoint.pt --episodes 64 --out results/validation_cartpole.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from policy_gradient import Network, make_env, rollout, returns_to_go, check_gradient


def finite_difference():
    # Compare autograd to a central finite difference through the actual actor.
    torch.manual_seed(123)
    net=Network('cartpole',2).double()
    states=torch.tensor([[.1,.2,-.1,.3],[-.2,.1,.2,-.1]],dtype=torch.float64)
    actions=torch.tensor([0,1])
    weights=torch.tensor([1.,.95],dtype=torch.float64)
    advantage=torch.tensor([2.1,-.4],dtype=torch.float64)
    def objective():
        return -(Categorical(logits=net(states)).log_prob(actions)*weights*advantage).sum()
    objective().backward()
    param=net.layers[-1].weight
    analytic=float(param.grad[0,0])
    with torch.no_grad():
        original=float(param[0,0]); eps=1e-5
        param[0,0]=original+eps; plus=float(objective())
        param[0,0]=original-eps; minus=float(objective())
        param[0,0]=original
    numeric=(plus-minus)/(2*eps)
    error=abs(analytic-numeric)
    assert error<1e-7,(analytic,numeric)
    return dict(autograd=analytic,central_difference=numeric,absolute_error=error)


def episode_gradient(policy,value,states,actions,rewards,gamma,use_baseline):
    policy.zero_grad(set_to_none=True)
    target=returns_to_go(rewards,gamma)
    for start in range(0,len(actions),256):
        sl=slice(start,start+256)
        x=torch.as_tensor(states[sl])
        with torch.no_grad():
            b=value(x).squeeze(-1) if use_baseline else torch.zeros(len(x))
        adv=torch.as_tensor(target[sl])-b
        weight=torch.tensor(gamma**np.arange(start,min(start+256,len(actions))),dtype=torch.float32)
        logp=Categorical(logits=policy(x)).log_prob(torch.tensor(actions[sl]))
        (weight*adv*logp).sum().backward()
    return torch.cat([p.grad.flatten() for p in policy.parameters()]).detach().numpy().copy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--game',choices=['cartpole','pong'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=64)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.episodes<3: p.error('Require at least 3 independent trajectories')
    torch.set_num_threads(1)
    result=dict(analytic_score_function=check_gradient(torch.device('cpu')),
                network_finite_difference=finite_difference())
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['value'] is None: p.error('Checkpoint must include a value baseline')
    policy=Network(args.game,2); policy.load_state_dict(checkpoint['policy']); policy.eval()
    value=Network(args.game,1); value.load_state_dict(checkpoint['value']); value.eval()
    torch.manual_seed(918273); np.random.seed(918273)
    env=make_env(args.game)
    gamma=.95 if args.game=='cartpole' else .99
    plain=[];baseline=[];scores=[]
    for e in range(args.episodes):
        states,actions,rewards,_=rollout(env,policy,args.game,torch.device('cpu'),8_000_000+e)
        plain.append(episode_gradient(policy,value,states,actions,rewards,gamma,False))
        baseline.append(episode_gradient(policy,value,states,actions,rewards,gamma,True))
        scores.append(sum(rewards))
        print(f'Frozen-policy trajectory {e+1}/{args.episodes}',flush=True)
    env.close()
    plain=np.asarray(plain,dtype=np.float64);baseline=np.asarray(baseline,dtype=np.float64)
    # The trace of sample covariance measures total variance of the full actor gradient.
    var_plain=float(np.var(plain,axis=0,ddof=1).sum())
    var_base=float(np.var(baseline,axis=0,ddof=1).sum())
    degenerate=var_plain<1e-12
    # Paired bootstrap over trajectories. This is exploratory, especially for small n.
    rng=np.random.default_rng(123);ratios=[]
    for _ in range(300):
        ix=rng.integers(0,len(plain),len(plain))
        denominator=np.var(plain[ix],axis=0,ddof=1).sum()
        if denominator>1e-12:
            ratios.append(np.var(baseline[ix],axis=0,ddof=1).sum()/denominator)
    result.update(game=args.game,checkpoint_episode=checkpoint['episode'],trajectories=args.episodes,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  source_experiment=checkpoint.get('source_experiment',args.checkpoint.parent.name),
                  frozen_actor=True,held_out_seeds_start=8_000_000,full_actor_gradient=True,
                  plain_gradient_variance_trace=var_plain,baseline_gradient_variance_trace=var_base,
                  gradient_covariance_numerically_degenerate=degenerate,
                  baseline_to_plain_ratio=var_base/var_plain if not degenerate else None,
                  ratio_bootstrap_95_percentile=np.quantile(ratios,[.025,.975]).tolist() if ratios else None,
                  evaluation_reward_mean=float(np.mean(scores)),
                  interpretation=('Near-zero gradient covariance; a variance ratio is not meaningful.' if degenerate else
                                  'Ratio below 1 means lower empirical gradient variance at this frozen policy only; no universal guarantee.'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
