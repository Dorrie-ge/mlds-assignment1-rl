"""Measure discounted training objective separately from undiscounted game score."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from policy_gradient import Network,make_env
from pong_batch import collect


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    p.add_argument('--out',type=Path,default=root/'results/pong_discounted_objective.json')
    args=p.parse_args()
    torch.set_num_threads(1);device=torch.device(args.device)
    envs=[make_env('pong') for _ in range(8)]
    measurements={}
    for name in ['random','pong_batch_none','pong_batch_value']:
        model=Network('pong',2)
        if name!='random':
            model.load_state_dict(torch.load(root/f'results/{name}/policy.pt',map_location='cpu',weights_only=True))
        model.to(device);model.eval()
        torch.manual_seed(3141);np.random.seed(3141)
        records=[]
        for start in range(0,args.episodes,8):
            eps=collect(envs,model,device,range(6_000_000+start,6_000_000+min(start+8,args.episodes)),False,name=='random')
            records.extend(dict(score=sum(ep['rewards']),discounted_return=float(ep['returns'][0]),
                                steps=len(ep['rewards'])) for ep in eps)
        measurements[name]=records
        print(name,'mean discounted return',np.mean([r['discounted_return'] for r in records]),flush=True)
    for env in envs:env.close()
    summaries={}
    for name,records in measurements.items():
        returns=np.array([r['discounted_return'] for r in records])
        scores=np.array([r['score'] for r in records])
        j=dict(episodes=len(records),mean_discounted_return=float(returns.mean()),
               std_discounted_return=float(returns.std(ddof=1)),mean_score=float(scores.mean()))
        if name!='random':
            differences=returns-np.array([r['discounted_return'] for r in measurements['random']])
            se=differences.std(ddof=1)/np.sqrt(len(differences))
            j.update(discounted_improvement_over_random=float(differences.mean()),
                     approximate_ci95=[float(differences.mean()-1.96*se),float(differences.mean()+1.96*se)])
        summaries[name]=j
    result=dict(gamma=.99,held_out_environment_seed_start=6_000_000,episodes_per_policy=args.episodes,
                note='Separate exact full-episode objective diagnostic; not the main 500-episode score evaluation.',
                summaries=summaries,per_episode=measurements)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
