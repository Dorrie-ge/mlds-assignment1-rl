"""Independently confirm the selected actor lr=.001 with two fresh training seeds."""
import argparse,csv,json,os,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from audit_results import main as audit_all
from run_learning_rate_comparison import digest,git
ROOT=Path(__file__).resolve().parent.parent
STATUS=ROOT.parent/'pong_seed_validation_status.json'
STATE={'status':'preparing','runs':{}};LOCK=threading.Lock()
SELECTED='pong_motion_lr001_value'
NEW=[(1,'pong_motion_lr001_value_seed1'),(2,'pong_motion_lr001_value_seed2')]
ALL=[(0,SELECTED)]+NEW

def record(status,run=None,**extra):
    with LOCK:
        if run:STATE['runs'][run]=dict(status=status,**extra)
        else:STATE.update(status=status,**extra)
        STATE['updated_local_time']=time.strftime('%Y-%m-%d %H:%M:%S %Z')
        temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(STATE,indent=2));temp.replace(STATUS)
    print(status,run,extra,flush=True)

def analyze(manifest):
    assert digest(ROOT/manifest['training_code_snapshot'])==manifest['training_code_sha256']
    assert digest(ROOT/'src/policy_gradient.py')==manifest['imported_module_sha256']
    for filename,expected in manifest['selected_seed0_hashes'].items():assert digest(ROOT/f'results/{SELECTED}'/filename)==expected
    rows=[];training=[]
    for seed,name in ALL:
        directory=ROOT/f'results/{name}';s=json.loads((directory/'summary.json').read_text())
        checkpoint=torch.load(directory/'validation_checkpoint.pt',map_location='cpu',weights_only=True)
        assert s['episodes']==checkpoint['episode']==3000 and s['evaluation_episodes']==500
        expected=dict(lr=.001,value_lr=.0003,seed=seed,baseline='value',batch_episodes=8,max_grad_norm=5.,gamma=.99,representation='split_frame_difference',time_weighting='uniform',time_weighting_start_episode=1,device=manifest['device'],train_seed_start=20000000+seed*100000,evaluation_seed_start=4000000+seed*10000)
        for key,value in expected.items():assert s[key]==value,(name,key,s[key],value)
        assert checkpoint['config']['resume'] is None
        assert checkpoint['optimizer']['param_groups'][0]['lr']==.001
        assert checkpoint['value_optimizer']['param_groups'][0]['lr']==.0003
        for optimizer in ['optimizer','value_optimizer']:assert {int(x['step'].item()) for x in checkpoint[optimizer]['state'].values()}=={375}
        with (directory/'training.csv').open() as stream:train=list(csv.DictReader(stream))
        assert [int(x['episode']) for x in train]==list(range(1,3001))
        x=np.loadtxt(directory/'evaluation.csv',skiprows=1);random=np.loadtxt(directory/'random_evaluation.csv',skiprows=1)
        assert len(x)==len(random)==500
        assert np.isclose(x.mean(),s['mean']) and np.isclose(x.std(ddof=1),s['std_sample'])
        d=x-random;margin=1.96*d.std(ddof=1)/np.sqrt(500)
        rows.append(dict(seed=seed,training_episodes=3000,evaluation_episodes=500,evaluation_mean=float(x.mean()),evaluation_sample_sd=float(x.std(ddof=1)),positive_score_episodes=int((x>0).sum()),random_mean=float(random.mean()),mean_gain_over_random=float(d.mean()),paired_rollout_gain_ci95=[float(d.mean()-margin),float(d.mean()+margin)],result_directory=f'results/{name}',role='selection_run' if seed==0 else 'independent_confirmation'))
        training.append(train)
    means=np.array([x['evaluation_mean'] for x in rows]);confirmation=means[1:]
    result=dict(runs=rows,three_seed_mean=float(means.mean()),sample_sd_between_three_seed_means=float(means.std(ddof=1)),independent_confirmation_mean=float(confirmation.mean()),confirmation_seeds=[1,2],limitation='Seed 0 was used to select the learning rate; seeds 1 and 2 are independent confirmation runs. Three training seeds are a small sample. Between-seed variability is reported separately from the 500 rollout scores per model; these are not 1500 independent training runs. All models have the same fixed training budget. No new low-learning-rate models are included, so this tests reproducibility of the selected configuration, not a full learning-rate-by-seed interaction.')
    (ROOT/'results/pong_seed_validation.json').write_text(json.dumps(result,indent=2));audit_all()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row,train in zip(rows,training):axes[0].plot(np.arange(100,3001),np.convolve([float(x['reward']) for x in train],np.ones(100)/100,'valid'),label=f"seed {row['seed']}")
    axes[0].set(xlabel='Training episodes',ylabel='100-episode mean score',title='Actor lr=0.001');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].bar([str(x['seed']) for x in rows],means,color=['#94a3b8','#2563eb','#2563eb']);axes[1].set(xlabel='Training seed',ylabel='500-episode evaluation mean',title='Selection run (gray) and confirmation runs');axes[1].axhline(0,color='black',lw=.6);axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'results/pong_seed_validation.png',dpi=180);plt.close(fig)
    block=['\n### Independent training-seed confirmation at actor lr=0.001','', '| Training seed | Role | 500-episode mean ± sample SD |','|---|---|---:|']
    block += [f"| {x['seed']} | {x['role']} | {x['evaluation_mean']:.3f} ± {x['evaluation_sample_sd']:.3f} |" for x in rows]
    block += ['',f"Mean across three training-seed means: {means.mean():.3f}; sample SD between seed means: {means.std(ddof=1):.3f}.",f"The two independent confirmation runs average {confirmation.mean():.3f}.",'', '![Seed validation](results/pong_seed_validation.png)','', '[Raw results and scope](results/pong_seed_validation.json). Seed 0 was used for selection; seeds 1 and 2 were held out from that selection. Three training seeds remain a small sample. The PDF has not been regenerated.','']
    readme=ROOT/'EXPERIMENTS.md';readme.write_text(readme.read_text()+'\n'.join(block));return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['mps','cpu','cuda'],default='mps');parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    manifest_path=ROOT/'results/pong_seed_validation_manifest.json';assert not manifest_path.exists(),'Validation already exists: inspect its status; do not duplicate it'
    previous=json.loads((ROOT/'results/pong_learning_rate_manifest.json').read_text());source=ROOT/previous['training_code_snapshot']
    assert digest(source)==previous['training_code_sha256']
    assert digest(ROOT/'src/policy_gradient.py')==previous['imported_module_sha256']
    selected=ROOT/f'results/{SELECTED}';summary=json.loads((selected/'summary.json').read_text());assert summary['device']==args.device and summary['lr']==.001
    manifest=dict(training_code_snapshot=str(source.relative_to(ROOT)),training_code_sha256=digest(source),imported_module_sha256=digest(ROOT/'src/policy_gradient.py'),base_commit=git('rev-parse','HEAD').strip(),actor_lr=.001,critic_lr=.0003,episodes_per_group=3000,evaluation_episodes=500,selection_seed=0,confirmation_seeds=[1,2],device=args.device,initialization='Fresh networks and optimizers for seeds 1 and 2; seed 0 reused unchanged.',between_group_change='training seed, including initialization, sampling and the existing per-seed environment/evaluation seed ranges',selected_seed0_hashes={f:digest(selected/f) for f in ['summary.json','training.csv','evaluation.csv','random_evaluation.csv','validation_checkpoint.pt']},outputs=[f'results/{name}' for _,name in ALL])
    manifest_path.write_text(json.dumps(manifest,indent=2));record('training',selection_seed0_reused=f'results/{SELECTED}')
    def train(pair):
        seed,name=pair;out=ROOT/f'results/{name}';assert not out.exists(),out
        env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
        command=[sys.executable,'-u',str(source),'--baseline','value','--episodes','3000','--batch-episodes','8','--eval-episodes','500','--lr','.001','--value-lr','.0003','--max-grad-norm','5','--seed',str(seed),'--device',args.device,'--time-weighting','uniform','--representation','split_frame_difference','--out',str(out)]
        with (ROOT/f'logs/{name}.log').open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=env);record('running',run=name,pid=process.pid,seed=seed);code=process.wait()
        if code:raise RuntimeError(f'{name} exited {code}; inspect its log')
        record('complete',run=name)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(train,NEW))
    record('auditing');result=analyze(manifest)
    if args.publish:
        assert not git('diff','--cached','--name-only').strip(),'Existing staged work requires review'
        assert set(git('diff','--name-only').splitlines())<={'EXPERIMENTS.md','results/audit.json'},'Unexpected tracked changes require review'
        paths=['EXPERIMENTS.md','results/audit.json','results/pong_seed_validation_manifest.json','results/pong_seed_validation.json','results/pong_seed_validation.png']+[f'results/{name}' for _,name in NEW]+[f'logs/{name}.log' for _,name in NEW]
        git('diff','--check');git('add','--',*paths);git('commit','-m','Record independent Pong training-seed validation');git('push','origin','main');git('archive','--format=zip','--output=../mlds_assignment1_measured_runs.zip','HEAD')
    record('complete',results=result,commit=git('rev-parse','HEAD').strip())
if __name__=='__main__':
    try:main()
    except Exception:record('failed',error=traceback.format_exc());raise
