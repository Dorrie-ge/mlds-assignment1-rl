"""Deterministic task-level sanity checks; these are hand-coded, not learned policies."""
from pathlib import Path
import json
import numpy as np
from policy_gradient import make_env
ROOT=Path(__file__).resolve().parent.parent
rows=[];trace=[]
for controller in ['always_left','angle_feedback']:
    env=make_env('cartpole')
    for episode in range(20):
        state,_=env.reset(seed=9500000+episode);total=0.
        for t in range(500):
            action=0 if controller=='always_left' else int(state[2]+.15*state[3]>0)
            before=state.copy();state,reward,terminated,truncated,_=env.step(action);total+=reward
            if episode==0 and t<10:trace.append(dict(controller=controller,t=t,state=before.tolist(),action=action,reward=float(reward),terminated=bool(terminated),truncated=bool(truncated)))
            if terminated or truncated:break
        assert 1<=total<=500 and total==t+1
        rows.append(dict(controller=controller,seed=9500000+episode,steps=t+1,total_reward=total,terminated=bool(terminated),truncated=bool(truncated)))
    env.close()
result=dict(purpose='Task-level environment, reward and terminal/truncation diagnostic. These deterministic hand-coded controllers are not trained-policy assignment results.',seed_start=9500000,episodes_per_controller=20,controllers=dict(always_left='Always action 0',angle_feedback='Action 1 iff pole_angle + 0.15*pole_angular_velocity > 0'),rows=rows,first_10_steps=trace,mean_scores={name:float(np.mean([x['total_reward'] for x in rows if x['controller']==name])) for name in ['always_left','angle_feedback']})
(ROOT/'results/cartpole_control_diagnostic.json').write_text(json.dumps(result,indent=2))
print(result['mean_scores'])
print('Time-limit truncations:',sum(x['truncated'] for x in rows))
