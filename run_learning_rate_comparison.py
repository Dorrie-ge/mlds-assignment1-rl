"""Compare actor learning rates at a fixed budget, reusing the measured low-rate control."""
import argparse,csv,hashlib,json,os,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from audit_results import main as audit_all
ROOT=Path(__file__).resolve().parent
STATUS=ROOT.parent/'pong_learning_rate_run_status.json'
LOCK=threading.Lock();STATE={'status':'preparing','runs':{}}
CONTROL='pong_repr_motion_value'
NEW=[(.0003,'pong_motion_lr0003_value'),(.001,'pong_motion_lr001_value')]
ALL=[(.0001,CONTROL)]+NEW

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT)
def record(status,run=None,**extra):
    with LOCK:
        if run:STATE['runs'][run]=dict(status=status,**extra)
        else:STATE.update(status=status,**extra)
        STATE['updated_local_time']=time.strftime('%Y-%m-%d %H:%M:%S %Z')
        temporary=STATUS.with_suffix('.tmp');temporary.write_text(json.dumps(STATE,indent=2));temporary.replace(STATUS)
    print(status,run,extra,flush=True)

def analyze(manifest):
    source=ROOT/manifest['training_code_snapshot']
    assert digest(source)==manifest['training_code_sha256']
    assert digest(ROOT/'policy_gradient.py')==manifest['imported_module_sha256']
    for filename,expected in manifest['control_hashes'].items():assert digest(ROOT/f'results/{CONTROL}'/filename)==expected
    rows=[];scores=[];training=[]
    with (ROOT/f'results/{CONTROL}/training.csv').open() as stream:control_training=list(csv.DictReader(stream))
    reference_random=np.loadtxt(ROOT/f'results/{CONTROL}/random_evaluation.csv',skiprows=1)
    for rate,name in ALL:
        directory=ROOT/f'results/{name}'
        s=json.loads((directory/'summary.json').read_text())
        c=torch.load(directory/'validation_checkpoint.pt',map_location='cpu',weights_only=True)
        assert s['episodes']==c['episode']==3000 and s['evaluation_episodes']==500
        for key,value in dict(lr=rate,value_lr=.0003,seed=0,baseline='value',batch_episodes=8,max_grad_norm=5.,gamma=.99,representation='split_frame_difference',time_weighting='uniform',time_weighting_start_episode=1,device=manifest['device'],train_seed_start=20000000,evaluation_seed_start=4000000).items():assert s[key]==value,(name,key,s[key],value)
        assert c['config']['resume'] is None
        assert c['optimizer']['param_groups'][0]['lr']==rate
        assert c['value_optimizer']['param_groups'][0]['lr']==.0003
        for optimizer in ['optimizer','value_optimizer']:assert {int(x['step'].item()) for x in c[optimizer]['state'].values()}=={375}
        with (directory/'training.csv').open() as stream:train=list(csv.DictReader(stream))
        assert [int(x['episode']) for x in train]==list(range(1,3001))
        fields=['episode','reward','discounted_return','length','entropy','right_fraction','points_won','first_point_step']
        assert all(np.isclose(float(a[k]),float(b[k]),rtol=1e-5,atol=1e-6) for a,b in zip(train[:8],control_training[:8]) for k in fields),'First-batch statistics differ before learning-rate effects'
        score=np.loadtxt(directory/'evaluation.csv',skiprows=1);random=np.loadtxt(directory/'random_evaluation.csv',skiprows=1)
        assert len(score)==len(random)==500 and np.array_equal(random,reference_random)
        assert np.isclose(score.mean(),s['mean']) and np.isclose(score.std(ddof=1),s['std_sample'])
        scores.append(score);training.append(train)
        rows.append(dict(actor_lr=rate,evaluation_mean=float(score.mean()),evaluation_sample_sd=float(score.std(ddof=1)),training_episodes=3000,evaluation_episodes=500,result_directory=f'results/{name}',reused_control=name==CONTROL))
    differences=[]
    for row,score in zip(rows[1:],scores[1:]):
        d=score-scores[0];margin=1.96*d.std(ddof=1)/np.sqrt(500)
        differences.append(dict(actor_lr=row['actor_lr'],mean_gain_vs_control=float(d.mean()),approximate_paired_ci95=[float(d.mean()-margin),float(d.mean()+margin)]))
    result=dict(runs=rows,differences=differences,limitation='One training seed; confidence intervals cover evaluation-rollout variability only. Comparing several learning rates and selecting the best requires independent confirmation. All new policies start fresh and the measured low-rate control is reused unchanged.')
    (ROOT/'results/pong_learning_rate_comparison.json').write_text(json.dumps(result,indent=2))
    audit_all()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for row,train in zip(rows,training):ax.plot(np.arange(100,3001),np.convolve([float(x['reward']) for x in train],np.ones(100)/100,'valid'),label=f"actor lr={row['actor_lr']:g}")
    ax.set(xlabel='Training episodes',ylabel='100-episode mean score',title='Fixed-budget actor learning-rate comparison');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/pong_learning_rate_comparison.png',dpi=180);plt.close(fig)
    block=['\n### Measured fixed-budget actor learning-rate comparison','', '| Actor learning rate | 500-episode mean ± sample SD |','|---|---:|']
    block += [f"| {x['actor_lr']:g} | {x['evaluation_mean']:.3f} ± {x['evaluation_sample_sd']:.3f} |" for x in rows]
    block += ['', '![Learning-rate comparison](results/pong_learning_rate_comparison.png)','', '[Raw comparison and paired intervals](results/pong_learning_rate_comparison.json).', 'The unchanged 0.0001 control is reused; the two new policies start fresh. This is a one-seed exploratory comparison. The PDF has not been regenerated.','']
    readme=ROOT/'README.md';readme.write_text(readme.read_text()+'\n'.join(block))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['mps','cpu','cuda'],default='mps');parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    manifest_path=ROOT/'results/pong_learning_rate_manifest.json'
    assert not manifest_path.exists(),'Existing learning-rate batch: inspect its status; do not duplicate it'
    source_manifest=json.loads((ROOT/'results/pong_representation_manifest.json').read_text());source=ROOT/source_manifest['training_code_snapshot']
    assert digest(source)==source_manifest['training_code_sha256']
    control=ROOT/f'results/{CONTROL}';control_summary=json.loads((control/'summary.json').read_text())
    assert control_summary['device']==args.device
    manifest=dict(training_code_snapshot=str(source.relative_to(ROOT)),training_code_sha256=digest(source),imported_module_sha256=digest(ROOT/'policy_gradient.py'),base_commit=git('rev-parse','HEAD').strip(),episodes_per_group=3000,evaluation_episodes=500,baseline='value',representation='split_frame_difference',time_weighting='uniform',actor_learning_rates=[x[0] for x in ALL],critic_learning_rate=.0003,seed=0,batch_episodes=8,max_grad_norm=5.,device=args.device,initialization='Fresh actor, critic and Adam states for new rates; existing fresh-run control reused unchanged.',only_between_group_change='actor learning rate',control_directory=f'results/{CONTROL}',control_hashes={name:digest(control/name) for name in ['summary.json','training.csv','evaluation.csv','random_evaluation.csv','validation_checkpoint.pt']},outputs=[f'results/{name}' for _,name in ALL])
    manifest_path.write_text(json.dumps(manifest,indent=2));record('training',control_reused=f'results/{CONTROL}')
    def train(pair):
        rate,name=pair;out=ROOT/f'results/{name}';assert not out.exists(),out
        env=dict(os.environ);env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        command=[sys.executable,'-u',str(source),'--baseline','value','--episodes','3000','--batch-episodes','8','--eval-episodes','500','--lr',str(rate),'--value-lr','.0003','--max-grad-norm','5','--seed','0','--device',args.device,'--time-weighting','uniform','--representation','split_frame_difference','--out',str(out)]
        with (ROOT/f'logs/{name}.log').open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=env);record('running',run=name,pid=process.pid,actor_lr=rate);code=process.wait()
        if code:raise RuntimeError(f'{name} exited {code}; inspect its log')
        record('complete',run=name)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(train,NEW))
    record('auditing');result=analyze(manifest)
    if args.publish:
        assert not git('diff','--cached','--name-only').strip(),'Existing staged changes need review'
        assert set(git('diff','--name-only').splitlines())<={'README.md','results/audit.json'},'Unexpected tracked changes need review'
        paths=['README.md','results/audit.json','results/pong_learning_rate_manifest.json','results/pong_learning_rate_comparison.json','results/pong_learning_rate_comparison.png']+[f'results/{name}' for _,name in NEW]+[f'logs/{name}.log' for _,name in NEW]
        git('diff','--check');git('add','--',*paths);git('commit','-m','Record fixed-budget Pong actor learning-rate comparison');git('push','origin','main');git('archive','--format=zip','--output=../mlds_assignment1_measured_runs.zip','HEAD')
    record('complete',results=result,commit=git('rev-parse','HEAD').strip())
if __name__=='__main__':
    try:main()
    except Exception:record('failed',error=traceback.format_exc());raise
