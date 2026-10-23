"""Run a fixed-budget, fresh-initialization comparison of two Pong inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
import torch
from audit_results import main as audit_all

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT.parent / 'pong_representation_run_status.json'
LOCK = threading.Lock()
STATE = {'status': 'preparing', 'runs': {}}
MODES = ['stacked_frames', 'split_frame_difference']
NAMES = ['pong_repr_stack_value', 'pong_repr_motion_value']


def record(status, mode=None, **extra):
    with LOCK:
        if mode is None:
            STATE.update(status=status, **extra)
        else:
            STATE['runs'][mode] = dict(status=status, **extra)
        STATE['updated_local_time'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
        temp = STATUS.with_suffix('.tmp')
        temp.write_text(json.dumps(STATE, indent=2))
        temp.replace(STATUS)
    print(status, mode, extra, flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def summarize(episodes):
    rows = []
    for mode, name in zip(MODES, NAMES):
        directory = ROOT / 'results' / name
        summary = json.loads((directory / 'summary.json').read_text())
        checkpoint = torch.load(directory / 'validation_checkpoint.pt', map_location='cpu', weights_only=True)
        assert summary['episodes'] == checkpoint['episode'] == episodes
        assert summary['representation'] == mode
        assert summary['time_weighting'] == 'uniform' and summary['time_weighting_start_episode'] == 1
        for optimizer in ['optimizer', 'value_optimizer']:
            assert {int(x['step'].item()) for x in checkpoint[optimizer]['state'].values()} == {(episodes+7)//8}
        score = np.loadtxt(directory / 'evaluation.csv', skiprows=1)
        assert len(score) == 500 and np.isclose(score.mean(), summary['mean'])
        assert np.isclose(score.std(ddof=1), summary['std_sample'])
        rows.append(dict(representation=mode, episodes=episodes, mean=float(score.mean()),
                         sample_sd=float(score.std(ddof=1)), result_directory=f'results/{name}'))
    a, b = [np.loadtxt(ROOT / f'results/{name}/evaluation.csv', skiprows=1) for name in NAMES]
    difference = b-a
    margin = 1.96*difference.std(ddof=1)/np.sqrt(500)
    result = dict(runs=rows, motion_minus_stack=float(difference.mean()),
                  approximate_paired_ci95=[float(difference.mean()-margin), float(difference.mean()+margin)],
                  limitation='One training seed. Both models start fresh with uniform actor time weights. Paired intervals describe evaluation-rollout variability, not training-seed variability.')
    (ROOT / 'results/pong_representation_comparison.json').write_text(json.dumps(result, indent=2))
    audit_all()
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, name in zip(MODES, NAMES):
        train = np.genfromtxt(ROOT / f'results/{name}/training.csv', delimiter=',', names=True)
        assert len(train) == episodes
        ax.plot(train['episode'][99:], np.convolve(train['reward'], np.ones(100)/100, 'valid'), label=mode)
    ax.set(xlabel='Training episodes', ylabel='100-episode mean game score', title='Fresh models: input representation comparison')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'results/pong_representation_comparison.png', dpi=180)
    plt.close(fig)
    readme = ROOT / 'EXPERIMENTS.md'
    block = ['\n### Measured fixed-budget input comparison', '',
             '| Input | Training episodes | 500-episode evaluation mean ± sample SD |', '|---|---:|---:|']
    block += [f"| {x['representation']} | {episodes} | {x['mean']:.3f} ± {x['sample_sd']:.3f} |" for x in rows]
    block += ['', '![Input comparison](results/pong_representation_comparison.png)', '',
              '[Raw comparison and uncertainty](results/pong_representation_comparison.json).',
              'Both models start fresh; only the input encoding differs. Uniform time weights, the CNN,',
              'value baseline, learning rates, batch size and seeds match. The PDF has not been regenerated.', '']
    readme.write_text(readme.read_text()+'\n'.join(block))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--wait-delivery', type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.episodes < 100:
        parser.error('Use at least 100 episodes for this comparison; use pong_batch.py for smoke tests')
    snapshot = ROOT / 'results/pong_representation_training_source.txt'
    manifest_path = ROOT / 'results/pong_representation_manifest.json'
    assert not manifest_path.exists(), 'Existing comparison: inspect its status instead of duplicating it'
    snapshot.write_bytes((ROOT / 'src/pong_batch.py').read_bytes())
    manifest = dict(training_code_snapshot=str(snapshot.relative_to(ROOT)), training_code_sha256=digest(snapshot),
                    imported_module_sha256=digest(ROOT / 'src/policy_gradient.py'), episodes_per_group=args.episodes,
                    evaluation_episodes=500, initialization='fresh for both models; no checkpoint transfer',
                    only_between_group_change='input representation', baseline='value', time_weighting='uniform',
                    seed=0, lr=.0001, value_lr=.0003, batch_episodes=8, max_grad_norm=5., device=args.device,
                    representations=MODES, outputs=[f'results/{x}' for x in NAMES])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if args.wait_delivery:
        record('waiting_for_current_batch', episodes_per_group=args.episodes)
        deadline = time.monotonic()+14400
        while True:
            status = json.loads(args.wait_delivery.read_text())['status']
            if status == 'complete':
                break
            if status == 'failed':
                raise RuntimeError('Previous batch delivery failed; inspect its status before continuing')
            if time.monotonic() > deadline:
                raise TimeoutError('Previous batch did not complete within four hours')
            time.sleep(10)
    assert digest(snapshot) == manifest['training_code_sha256']
    assert digest(ROOT / 'src/policy_gradient.py') == manifest['imported_module_sha256']
    record('training')
    def train(pair):
        mode, name = pair
        out = ROOT / 'results' / name
        assert not out.exists(), f'Existing output directory: {out}'
        log = ROOT / 'logs' / f'{name}.log'
        command = [sys.executable, '-u', str(snapshot), '--baseline', 'value', '--episodes', str(args.episodes),
                   '--batch-episodes', '8', '--eval-episodes', '500', '--lr', '.0001', '--value-lr', '.0003',
                   '--max-grad-norm', '5', '--seed', '0', '--device', args.device,
                   '--time-weighting', 'uniform', '--representation', mode, '--out', str(out)]
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH', '')
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            record('running', mode=mode, pid=process.pid)
            code = process.wait()
        if code:
            raise RuntimeError(f'{mode} exited {code}; inspect {log}')
        record('complete', mode=mode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(train, zip(MODES, NAMES)))
    record('auditing')
    result = summarize(args.episodes)
    if args.publish:
        assert not git('diff', '--cached', '--name-only').strip(), 'Existing staged work requires review'
        assert set(git('diff', '--name-only').splitlines()) <= {'EXPERIMENTS.md', 'results/audit.json'}
        paths = ['EXPERIMENTS.md', 'results/audit.json', 'results/pong_representation_comparison.json',
                 'results/pong_representation_comparison.png', 'results/pong_representation_manifest.json',
                 'results/pong_representation_training_source.txt']
        paths += [f'results/{name}' for name in NAMES]+[f'logs/{name}.log' for name in NAMES]
        git('diff', '--check')
        git('add', '--', *paths)
        git('commit', '-m', 'Record fixed-budget Pong input representation comparison')
        git('push', 'origin', 'main')
        git('archive', '--format=zip', '--output=../mlds_assignment1_measured_runs.zip', 'HEAD')
    record('complete', results=result, commit=git('rev-parse', 'HEAD').strip())


if __name__ == '__main__':
    try:
        main()
    except Exception:
        record('failed', error=traceback.format_exc())
        raise
