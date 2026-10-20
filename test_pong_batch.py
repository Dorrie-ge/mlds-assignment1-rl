"""Numerically verify the batched, chunked actor update against a dense reference."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from policy_gradient import Network,returns_to_go
from pong_batch import update,unpack

torch.set_num_threads(1)
torch.manual_seed(31)
rng=np.random.default_rng(31)
policy=Network('pong',2)
value=Network('pong',1)
episodes=[]
loss=0.
for length in (270,7):
    states=rng.integers(0,2,(length,2,80,80),dtype=np.uint8)
    packed=np.packbits(states.reshape(length,-1),axis=1)
    assert np.array_equal(unpack(packed),states)
    actions=rng.integers(0,2,length).tolist()
    rewards=rng.choice([-1.,0.,1.],length).tolist()
    target=returns_to_go(rewards,.99)
    x=torch.as_tensor(states,dtype=torch.float32)
    with torch.no_grad():baseline=value(x).squeeze(-1)
    weight=torch.tensor(.99**np.arange(length),dtype=torch.float32)
    loss=loss-(weight*Categorical(logits=policy(x)).log_prob(torch.tensor(actions))*(torch.tensor(target)-baseline)).sum()/2
    episodes.append(dict(packed=packed,actions=actions,rewards=rewards,returns=target))
expected=torch.autograd.grad(loss,list(policy.parameters()))
# lr=0 isolates the computed gradient; a large clipping bound keeps it unchanged.
opt=torch.optim.SGD(policy.parameters(),lr=0.)
vopt=torch.optim.SGD(value.parameters(),lr=.001)
update(policy,value,opt,vopt,episodes,torch.device('cpu'),1e10)
errors=[float((p.grad-g).abs().max()) for p,g in zip(policy.parameters(),expected)]
relative=float(torch.linalg.vector_norm(torch.cat([(p.grad-g).flatten() for p,g in zip(policy.parameters(),expected)]))/
               torch.linalg.vector_norm(torch.cat([g.flatten() for g in expected])))
assert max(errors)<2e-5 and relative<2e-5,(errors,relative)
result=dict(passed=True,episode_lengths=[270,7],crosses_chunk_boundary=True,
            reference='Dense average of full discounted episodic gradients, detached pre-fit critic',
            max_parameter_gradient_error=max(errors),relative_l2_error=relative)
out=Path(__file__).resolve().parent/'results/validation_pong_batch_gradient.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
