"""Audit completed 5,000-episode control and uniform-time warm-start experiments."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import numpy as np
import torch
from audit_results import main as audit_all

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows_out = []
    for kind, manifest_name in [('control', 'pong_budget5000'), ('uniform', 'pong_uniform5000')]:
        manifest = json.loads((ROOT / f'results/{manifest_name}_manifest.json').read_text())
        source = ROOT / manifest.get('training_code_snapshot', 'src/pong_batch.py')
        assert digest(source) == manifest['training_code_sha256']
        assert digest(ROOT / 'src/policy_gradient.py') == manifest['imported_training_module_sha256']
        for entry in manifest['runs']:
            old, new = ROOT / entry['source'], ROOT / entry['output']
            assert digest(old / 'checkpoint.pt') == entry['source_checkpoint_sha256']
            assert digest(old / 'training.csv') == entry['source_training_csv_sha256']
            assert (new / 'training.csv').read_bytes().startswith((old / 'training.csv').read_bytes())
            with (new / 'training.csv').open() as stream:
                train = list(csv.DictReader(stream))
            assert [int(x['episode']) for x in train] == list(range(1, 5001))
            summary = json.loads((new / 'summary.json').read_text())
            original = json.loads((old / 'summary.json').read_text())
            for key in ['baseline', 'gamma', 'lr', 'seed', 'batch_episodes', 'value_lr',
                        'max_grad_norm', 'device', 'train_seed_start', 'evaluation_seed_start']:
                assert summary[key] == original[key], key
            assert summary['episodes'] == 5000 and summary['evaluation_episodes'] == 500
            if kind == 'uniform':
                assert summary['time_weighting'] == 'uniform'
                assert summary['time_weighting_start_episode'] == 3001
                control = ROOT / f"results/pong_budget5000_{summary['baseline']}"
                with (control / 'training.csv').open() as stream:
                    control_train = list(csv.DictReader(stream))
                fields = ['episode', 'reward', 'discounted_return', 'length',
                          'entropy', 'right_fraction', 'points_won', 'first_point_step']
                paired = all(np.isclose(float(a[key]), float(b[key]), rtol=1e-5, atol=1e-6)
                             for a, b in zip(control_train[3000:3008], train[3000:3008])
                             for key in fields)
                assert paired, 'Recorded first-batch statistics differ before the first changed update'
                pairing = dict(passed=True, baseline=summary['baseline'], episodes=[3001, 3008],
                               checked_fields=fields,
                               note='Recorded first-batch rollout statistics match before the first differently weighted actor update; gradient_norm is intentionally excluded.')
                (new / 'first_batch_pairing_check.json').write_text(json.dumps(pairing, indent=2))
            else:
                assert summary.get('time_weighting', 'episode_discount') == 'episode_discount'
            checkpoint = torch.load(new / 'validation_checkpoint.pt', map_location='cpu', weights_only=True)
            assert checkpoint['episode'] == 5000
            for optimizer in ['optimizer', 'value_optimizer']:
                if checkpoint[optimizer] is not None:
                    assert {int(x['step'].item()) for x in checkpoint[optimizer]['state'].values()} == {625}
            scores = np.loadtxt(new / 'evaluation.csv', skiprows=1)
            random = np.loadtxt(new / 'random_evaluation.csv', skiprows=1)
            assert len(scores) == len(random) == 500
            assert np.isclose(scores.mean(), summary['mean'])
            assert np.isclose(scores.std(ddof=1), summary['std_sample'])
            assert np.array_equal(random, np.loadtxt(old / 'random_evaluation.csv', skiprows=1))
            rows_out.append(dict(kind=kind, baseline=summary['baseline'],
                                 mean=float(scores.mean()), sample_sd=float(scores.std(ddof=1)),
                                 evaluation_episodes=500, source_3000_mean=original['mean'],
                                 total_training_steps=sum(int(x['length']) for x in train),
                                 final100_training_mean=float(np.mean([float(x['reward']) for x in train[-100:]])),
                                 final100_training_entropy=float(np.mean([float(x['entropy']) for x in train[-100:]])),
                                 result_directory=str(new.relative_to(ROOT)),
                                 original_csv_exact_prefix=True, optimizer_steps=625))
            local_log = ROOT.parents[1] / 'work/logs' / f'{new.name}.log'
            if local_log.exists():
                shutil.copyfile(local_log, ROOT / 'logs' / local_log.name)
    differences = []
    for mode in ['none', 'value']:
        a = np.loadtxt(ROOT / f'results/pong_budget5000_{mode}/evaluation.csv', skiprows=1)
        b = np.loadtxt(ROOT / f'results/pong_uniform5000_{mode}/evaluation.csv', skiprows=1)
        d = b - a
        margin = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        differences.append(dict(baseline=mode, uniform_minus_control=float(d.mean()),
                                approximate_paired_ci95=[float(d.mean()-margin), float(d.mean()+margin)]))
    result = dict(runs=rows_out, paired_differences=differences,
                  limitation='One training seed; intervals describe held-out rollout variability, not training-seed uncertainty. Uniform weighting begins at episode 3001 after a discounted-objective warm start, with inherited optimizer state.',
                  objective_note='Uniform time weighting removes only the actor outer gamma**t. Full-episode discounted reward-to-go at gamma=.99 is retained; this is a changed policy-gradient surrogate.')
    (ROOT / 'results/pong_time_weighting_comparison.json').write_text(json.dumps(result, indent=2))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ['none', 'value']):
        for kind, prefix in [('Original weighting', 'pong_budget5000'), ('Uniform after episode 3000', 'pong_uniform5000')]:
            train = np.genfromtxt(ROOT / f'results/{prefix}_{mode}/training.csv', delimiter=',', names=True)
            ax.plot(train['episode'][99:], np.convolve(train['reward'], np.ones(100)/100, 'valid'), label=kind)
        ax.axvline(3000, color='gray', linestyle='--')
        ax.set(xlabel='Training episodes', title=f'Baseline: {mode}', xlim=(0, 5000))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('100-episode mean game score')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/pong_time_weighting_comparison.png', dpi=180)
    plt.close(fig)
    audit_all()
    start, end = '<!-- TIME WEIGHTING RESULTS START -->', '<!-- TIME WEIGHTING RESULTS END -->'
    block = [start, '### Measured Pong time-weighting comparison', '',
             'Both branches start from the same 3,000-episode checkpoint and reach 5,000 episodes.',
             'Uniform weighting starts at episode 3,001; all other hyperparameters and inherited optimizer states match.', '',
             '| Time weighting | Baseline | 500-episode evaluation mean ± sample SD |',
             '|---|---|---:|']
    for row in rows_out:
        block.append(f"| {row['kind']} | {row['baseline']} | {row['mean']:.3f} ± {row['sample_sd']:.3f} |")
    block += ['', '![Training comparison](results/pong_time_weighting_comparison.png)', '',
              '[Audit and paired differences](results/pong_time_weighting_comparison.json).',
              'This is one training seed and a warm-start experiment. Uniform time weighting changes',
              'the gradient surrogate; gamma=0.99 remains inside reward-to-go. The existing PDF has not been regenerated.', end]
    readme = ROOT / 'EXPERIMENTS.md'
    text = readme.read_text()
    text = text.replace('### Pong outer time-weighting ablation (running)',
                        '### Pong outer time-weighting ablation (completed)')
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + '\n'.join(block) + after
    else:
        text += '\n\n' + '\n'.join(block) + '\n'
    readme.write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
