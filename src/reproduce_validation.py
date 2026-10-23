"""Repeat analytic and frozen-policy validation using committed weights; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=root/'validation_reproduced')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'results/validation_checkpoints.json').read_text())
    for row in manifest:
        path=root/row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
    print('All committed validation-capsule hashes match.',flush=True)
    subprocess.run([sys.executable,str(root/'src/policy_gradient.py'),'--test-only'],check=True)
    subprocess.run([sys.executable,str(root/'src/test_pong_batch.py')],check=True)
    for game,name,count in [('cartpole','cartpole_value_lr0001',64),('pong','pong_batch_value',16)]:
        subprocess.run([sys.executable,str(root/'src/validate_policy_gradient.py'),'--game',game,
                        '--checkpoint',str(root/f'results/{name}/validation_checkpoint.pt'),
                        '--episodes',str(count),'--out',str(args.out/f'{game}.json')],check=True)
    print('Completed frozen-policy validation without retraining. Results:',args.out)


if __name__=='__main__':main()
