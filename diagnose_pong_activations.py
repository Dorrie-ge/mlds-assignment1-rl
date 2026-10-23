from pathlib import Path
import sys,io,json,hashlib,time
import numpy as np
import torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from policy_gradient import Network,make_env,preprocess
from pong_batch import encode_state
torch.set_num_threads(1)
e=make_env('pong');obs,_=e.reset(seed=9000000);previous=preprocess(obs);states=[];rng=np.random.default_rng(2026)
for step in range(256):
 current=preprocess(obs);states.append(encode_state(previous,current,'split_frame_difference'));previous=current
 obs,_,done,truncated,_=e.step(int(rng.choice([2,3])))
 if done or truncated:obs,_=e.reset(seed=9000001+step);previous=preprocess(obs)
e.close();x=torch.tensor(np.stack(states));results=[]
for name in ['pong_repr_motion_value','pong_motion_lr0003_value','pong_motion_lr001_value']:
 p=R/f'results/{name}/validation_checkpoint.pt'
 for attempt in range(5):
  try:
   payload=p.read_bytes();c=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True);break
  except (RuntimeError,EOFError):time.sleep(.1)
 row=dict(run=name,checkpoint_episode=c['episode'],checkpoint_sha256=hashlib.sha256(payload).hexdigest(),actor_lr=c['config']['lr'],networks={})
 for role,key,n in [('actor','policy',2),('critic','value',1)]:
  model=Network('pong',n);model.load_state_dict(c[key]);z=x;layers=[]
  with torch.no_grad():
   for i,layer in enumerate(model.layers):
    z=layer(z)
    if isinstance(layer,torch.nn.ReLU):layers.append(dict(layer=i,positive_fraction=float((z>0).float().mean()),output_std=float(z.std())))
   out=dict(relu_layers=layers)
   if role=='actor':
    probs=z.softmax(-1)[:,0];out.update(action2_probability_mean=float(probs.mean()),action2_probability_std_across_states=float(probs.std()))
   else:out.update(prediction_mean=float(z.mean()),prediction_std=float(z.std()))
  row['networks'][role]=out
 results.append(row)
out=dict(purpose='Network-health diagnostic on 256 independent random-policy observations. Committed 3000-episode models are used. This is not a score evaluation.',environment_seed=9000000,action_rng_seed=2026,results=results)
(R/'results/pong_endpoint_activations.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
