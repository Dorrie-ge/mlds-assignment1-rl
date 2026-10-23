from pathlib import Path
import sys,csv,json,time
import numpy as np
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R/'src'))
from policy_gradient import make_env,preprocess

def locate(frame):
    paddle=np.argwhere(frame[:,70:72]>0)
    p=float(paddle[:,0].mean()) if len(paddle) else 40.
    # Exclude the two paddles; the game ball occupies the remaining playfield.
    ball=np.argwhere(frame[:,10:68]>0)
    y=float(ball[:,0].mean()) if 0<len(ball)<=8 else None
    return p,y

start=time.perf_counter();rows=[];trace=[]
for mode in ['tracking','random']:
    e=make_env('pong');rng=np.random.default_rng(21)
    for episode in range(10):
        obs,_=e.reset(seed=8000000+episode);score=0.;detected=0;points=0
        for t in range(20000):
            f=preprocess(obs);p,b=locate(f);detected+=b is not None
            action=int(rng.choice([2,3])) if mode=='random' else (2 if (b if b is not None else 40.)<p else 3)
            obs,reward,terminated,truncated,_=e.step(action);score+=reward;points+=int(reward!=0)
            if mode=='tracking' and episode==0 and (t<200 or reward):trace.append(dict(t=t,paddle_y=p,ball_y=b,action=action,reward=float(reward)))
            if terminated or truncated:break
        rows.append(dict(policy=mode,episode=episode,seed=8000000+episode,score=score,steps=t+1,ball_detected_fraction=detected/(t+1),terminated=bool(terminated),truncated=bool(truncated),diagnostic_cap_reached=not(terminated or truncated)))
        print(mode,episode,score,t+1,detected/(t+1),flush=True)
    e.close()
out=dict(purpose='Environment/input/action diagnostic only. Tracking controller is hand-coded, not a learned policy or assignment model result.',actions=[2,3],seed_start=8000000,episodes_per_policy=10,rows=rows,tracking_first_episode_trace=trace,elapsed_seconds=time.perf_counter()-start)
checks=[]
e=make_env('pong')
for action in [2,3]:
    obs,_=e.reset(seed=8000000);ys=[]
    for t in range(4):
        ys.append(locate(preprocess(obs))[0]);obs,*_=e.step(action)
    checks.append(dict(action=action,paddle_y=ys,delta=ys[-1]-ys[0]))
e.close()
assert checks[0]['delta']<0 and checks[1]['delta']>0
out['action_response']=checks
out['random_action_seed']=21
out['mean_scores']={mode:float(np.mean([x['score'] for x in rows if x['policy']==mode])) for mode in ['tracking','random']}
(R/'results/pong_control_diagnostic.json').write_text(json.dumps(out,indent=2))
