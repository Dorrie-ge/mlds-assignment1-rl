"""Batched on-policy REINFORCE for the assignment's unmodified ALE/Pong-v5.

Default: same full-episode discounted objective as policy_gradient.py.
An explicit uniform time-weighting ablation keeps discounted reward-to-go but
removes the outer gamma**t factor; this changes the policy-gradient surrogate.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from policy_gradient import Network, make_env, preprocess, returns_to_go, figures, check_gradient


def unpack(packed):
    return np.unpackbits(np.asarray(packed),axis=1).reshape(-1,2,80,80).astype(np.float32)


def collect(envs,policy,device,seeds,store=True,random_policy=False):
    n=len(seeds)
    obs=[envs[i].reset(seed=int(seed))[0] for i,seed in enumerate(seeds)]
    previous=[preprocess(o) for o in obs]
    active=list(range(n))
    episodes=[dict(packed=[],actions=[],rewards=[],entropy=[],trace=[]) for _ in seeds]
    while active:
        current=[preprocess(obs[i]) for i in active]
        states=np.stack([np.stack((previous[i],frame)) for i,frame in zip(active,current)])
        with torch.no_grad():
            if random_policy:
                actions=np.random.randint(2,size=len(active))
                entropies=np.full(len(active),np.log(2.))
            else:
                distribution=Categorical(logits=policy(torch.as_tensor(states,device=device)))
                # Sample on CPU for consistent behavior across CPU/MPS backends.
                actions=torch.multinomial(distribution.probs.cpu(),1).squeeze(1).numpy()
                entropies=distribution.entropy().cpu().numpy()
        next_active=[]
        for row,(i,frame,action,entropy) in enumerate(zip(active,current,actions,entropies)):
            record=episodes[i]
            if store:record['packed'].append(np.packbits(states[row].astype(np.uint8)))
            record['actions'].append(int(action));record['entropy'].append(float(entropy))
            obs[i],reward,terminated,truncated,_=envs[i].step((2,3)[int(action)])
            record['rewards'].append(float(reward))
            if len(record['trace'])<10:
                record['trace'].append(dict(t=len(record['trace']),env_action=(2,3)[int(action)],
                                            reward=float(reward),entropy=float(entropy),
                                            terminated=bool(terminated),truncated=bool(truncated)))
            previous[i]=frame
            if not (terminated or truncated):next_active.append(i)
        active=next_active
    for record in episodes:
        if store:record['packed']=np.asarray(record['packed'],dtype=np.uint8)
        record['returns']=returns_to_go(record['rewards'],.99)
    return episodes


def update(policy,value,opt,vopt,episodes,device,max_norm,time_weighting='episode_discount'):
    if time_weighting not in ('episode_discount','uniform'):
        raise ValueError(f'Unknown time weighting: {time_weighting}')
    opt.zero_grad(set_to_none=True)
    for ep in episodes:
        length=len(ep['rewards'])
        for start in range(0,length,256):
            end=min(start+256,length);sl=slice(start,end)
            x=torch.as_tensor(unpack(ep['packed'][sl]),device=device)
            target=torch.as_tensor(ep['returns'][sl],device=device)
            with torch.no_grad():
                baseline=value(x).squeeze(-1) if value is not None else torch.zeros_like(target)
            # Uniform weighting changes only the outer time factor. The targets
            # remain full-episode gamma=.99 reward-to-go in both modes.
            weight=(torch.tensor(.99**np.arange(start,end),dtype=torch.float32,device=device)
                    if time_weighting=='episode_discount' else torch.ones_like(target))
            actions=torch.tensor(ep['actions'][sl],device=device)
            loss=-(weight*Categorical(logits=policy(x)).log_prob(actions)*(target-baseline)).sum()/len(episodes)
            loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),max_norm,error_if_nonfinite=True))
    opt.step()
    if value is not None:
        vopt.zero_grad(set_to_none=True)
        # One critic update after the actor: no fitting to the current batch before
        # the actor's detached baseline predictions have all been computed.
        for ep in episodes:
            length=len(ep['rewards'])
            for start in range(0,length,256):
                sl=slice(start,start+256)
                x=torch.as_tensor(unpack(ep['packed'][sl]),device=device)
                target=torch.as_tensor(ep['returns'][sl],device=device)
                loss=((value(x).squeeze(-1)-target)**2).sum()/(length*len(episodes))
                loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(),max_norm,error_if_nonfinite=True)
        vopt.step()
    return norm


def save(path,policy,value,opt,vopt,episode,args):
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    torch.save(dict(policy=policy.state_dict(),value=value.state_dict() if value is not None else None,
                    optimizer=opt.state_dict(),value_optimizer=vopt.state_dict() if vopt else None,
                    episode=episode,config=config,torch_rng=torch.get_rng_state()),path)


def evaluate(envs,policy,device,count,seed_start,random_policy=False):
    scores=[]
    for start in range(0,count,len(envs)):
        seeds=range(seed_start+start,seed_start+min(start+len(envs),count))
        episodes=collect(envs,policy,device,seeds,False,random_policy)
        scores.extend(sum(ep['rewards']) for ep in episodes)
        if len(scores)%100==0 or len(scores)==count:
            print(f"{'Random' if random_policy else 'Final'} evaluation {len(scores)}/{count}: {np.mean(scores):.3f}",flush=True)
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',choices=['none','value'],required=True)
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--batch-episodes',type=int,default=8)
    p.add_argument('--eval-episodes',type=int,default=500)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--value-lr',type=float,default=.0003)
    p.add_argument('--max-grad-norm',type=float,default=5.)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--time-weighting',choices=['episode_discount','uniform'],default='episode_discount')
    p.add_argument('--resume',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if min(args.episodes,args.batch_episodes)<1 or args.eval_episodes<2:p.error('Invalid episode count')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device)
    check_gradient(torch.device('cpu'))
    policy=Network('pong',2).to(device)
    torch.nn.init.orthogonal_(policy.layers[-1].weight,gain=.01)
    torch.nn.init.zeros_(policy.layers[-1].bias)
    # Preserve the actor sampling RNG when constructing the optional critic.
    with torch.random.fork_rng(devices=[]):
        value=Network('pong',1).to(device) if args.baseline=='value' else None
    if value is not None:
        torch.nn.init.zeros_(value.layers[-1].weight);torch.nn.init.zeros_(value.layers[-1].bias)
    opt=torch.optim.Adam(policy.parameters(),lr=args.lr)
    vopt=torch.optim.Adam(value.parameters(),lr=args.value_lr) if value is not None else None
    args.out.mkdir(parents=True,exist_ok=True)
    args.time_weighting_start_episode=1
    completed=0;training=[];started=time.perf_counter()
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=True)
        for field in ['baseline','batch_episodes','lr','value_lr','seed','max_grad_norm']:
            if checkpoint['config'][field]!=getattr(args,field):p.error(f'Resume changes {field}')
        policy.load_state_dict(checkpoint['policy']);opt.load_state_dict(checkpoint['optimizer'])
        if value is not None:
            value.load_state_dict(checkpoint['value']);vopt.load_state_dict(checkpoint['value_optimizer'])
        completed=checkpoint['episode'];torch.set_rng_state(checkpoint['torch_rng'].cpu())
        previous_weighting=checkpoint['config'].get('time_weighting','episode_discount')
        args.time_weighting_start_episode=(completed+1 if previous_weighting!=args.time_weighting
                                          else checkpoint['config'].get('time_weighting_start_episode',1))
        old=np.genfromtxt(args.out/'training.csv',delimiter=',',names=True)
        training=np.atleast_1d(old['reward']).tolist()
        if len(training)!=completed:p.error('CSV and checkpoint episode counts differ')
    elif (args.out/'training.csv').exists():p.error('Existing run: use --resume or another output directory')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    envs=[make_env('pong') for _ in range(args.batch_episodes)]
    with (args.out/'training.csv').open('a' if completed else 'w',newline='') as f:
        writer=csv.writer(f)
        if not completed:writer.writerow(['episode','reward','discounted_return','length','gradient_norm','entropy','right_fraction','points_won','first_point_step'])
        while completed<args.episodes:
            count=min(args.batch_episodes,args.episodes-completed)
            episodes=collect(envs,policy,device,range(20_000_000+args.seed*100000+completed,
                                                   20_000_000+args.seed*100000+completed+count))
            norm=update(policy,value,opt,vopt,episodes,device,args.max_grad_norm,args.time_weighting)
            for ep in episodes:
                completed+=1;training.append(sum(ep['rewards']))
                point_steps=np.flatnonzero(ep['rewards'])
                writer.writerow([completed,training[-1],float(ep['returns'][0]),len(ep['rewards']),norm,
                                 float(np.mean(ep['entropy'])),float(np.mean(np.array(ep['actions'])==0)),
                                 int(np.sum(np.array(ep['rewards'])>0)),int(point_steps[0]) if len(point_steps) else -1])
            f.flush()
            save(args.out/'checkpoint.pt',policy,value,opt,vopt,completed,args)
            print(f'Episodes {completed}: recent mean={np.mean(training[-100:]):.3f}; '
                  f'entropy={np.mean([np.mean(ep["entropy"]) for ep in episodes]):.4f}; grad={norm:.4g}',flush=True)
    save(args.out/'validation_checkpoint.pt',policy,value,opt,vopt,completed,args)
    torch.save(policy.state_dict(),args.out/'policy.pt')
    policy.eval()
    scores=evaluate(envs,policy,device,args.eval_episodes,4_000_000+args.seed*10000)
    random_scores=evaluate(envs,policy,device,args.eval_episodes,4_000_000+args.seed*10000,True)
    for env in envs:env.close()
    for name,values in [('evaluation',scores),('random_evaluation',random_scores)]:
        np.savetxt(args.out/f'{name}.csv',values,header='episode_reward',comments='')
    summary=dict(game='pong',env_id='ALE/Pong-v5',baseline=args.baseline,episodes=completed,
                 evaluation_episodes=args.eval_episodes,gamma=.99,lr=args.lr,seed=args.seed,
                 mean=float(np.mean(scores)),std_sample=float(np.std(scores,ddof=1)),
                 random_mean=float(np.mean(random_scores)),random_std_sample=float(np.std(random_scores,ddof=1)),
                 elapsed_seconds=time.perf_counter()-started,device=str(device),batch_episodes=args.batch_episodes,
                 max_grad_norm=args.max_grad_norm,value_lr=args.value_lr,
                 objective=('E[sum_t gamma**t*r_t]; full episode returns; average of per-episode score gradients'
                            if args.time_weighting=='episode_discount' else
                            'Uniform-time reward-to-go policy-gradient surrogate; gamma=.99 inside returns only; average of per-episode score gradients'),
                 time_weighting=args.time_weighting,time_weighting_start_episode=args.time_weighting_start_episode,
                 train_seed_start=20_000_000+args.seed*100000,evaluation_seed_start=4_000_000+args.seed*10000,
                 versions=dict(torch=torch.__version__,numpy=np.__version__),
                 limitation='One training seed; 500 stochastic evaluation episodes do not measure training-seed variability.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    figures(args.out,training,scores,random_scores)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
