"""Export small, portable actor/critic capsules and verify their weight identity."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent


def main():
    rows=[]
    names=['cartpole_value','cartpole_value_lr0001','pong_value','pong_value_early_diagnostic']
    names += [name for name in ['pong_batch_value'] if (ROOT/'results'/name/'checkpoint.pt').exists()]
    for name in names:
        src=ROOT/'results'/name/'checkpoint.pt'
        dst=src.with_name('validation_checkpoint.pt')
        if src.exists():
            state=torch.load(src,map_location='cpu',weights_only=True)
            capsule={key:state[key] for key in ['policy','value','episode']}
            capsule['source_experiment']=name
            if 'config' in state:capsule['config']=state['config']
            torch.save(capsule,dst)
            loaded=torch.load(dst,map_location='cpu',weights_only=True)
            for net in ['policy','value']:
                for key,value in state[net].items():
                    assert torch.equal(value.cpu(),loaded[net][key]),(name,net,key)
        loaded=torch.load(dst,map_location='cpu',weights_only=True)
        rows.append(dict(path=str(dst.relative_to(ROOT)),episode=loaded['episode'],
                         contains_actor=True,contains_critic=loaded['value'] is not None,
                         sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size))
    (ROOT/'results/validation_checkpoints.json').write_text(json.dumps(rows,indent=2))
    print('Validated actor/critic capsules:',len(rows))


if __name__=='__main__':main()
