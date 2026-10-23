"""Build the README measurement section directly from completed experiment files."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
START='<!-- MEASUREMENTS START -->'
END='<!-- MEASUREMENTS END -->'


def main():
    result=ROOT/'results'
    rows=[]
    config=json.loads((ROOT/'report_config.json').read_text())
    main_names=config['main_experiments']
    summaries={name:json.loads((result/name/'summary.json').read_text()) for name in main_names}
    lines=[START,'## Measured results','',
           'Focus: matched CartPole lr=0.001 and batched Pong lr=0.0001, seed 0. All evaluation budgets below are 500 episodes.','',
           '| Environment | Baseline | Training episodes | Mean reward | Sample SD | Random-policy mean |',
           '|---|---|---:|---:|---:|---:|']
    for name,j in summaries.items():
        lines.append(f"| {j['env_id']} | {j['baseline']} | {j['episodes']} | {j['mean']:.3f} | {j['std_sample']:.3f} | {j['random_mean']:.3f} |")
        rewards=np.loadtxt(result/name/'evaluation.csv',skiprows=1)
        assert len(rewards)==500
        assert abs(rewards.mean()-j['mean'])<1e-8
        assert abs(rewards.std(ddof=1)-j['std_sample'])<1e-8
    lines += ['',
        'The previous lr=0.01 failures are retained below. The CartPole plots use the same seed',
        'for both baselines rather than selecting different best seeds. Pong is retrained for',
        '1,000 episodes with trajectory averaging and a lower learning rate; performance is reported',
        'as measured, without claiming mastery. No unmeasured longer-run score is claimed.','',
        '### Historical failures','',
        '| Experiment | Episodes | Actor lr | Evaluation mean | Sample SD |','|---|---:|---:|---:|---:|']
    for name in ['cartpole_none','cartpole_value','pong_none','pong_value']:
        j=json.loads((result/name/'summary.json').read_text())
        lines.append(f"| {name} | {j['episodes']} | {j['lr']} | {j['mean']:.3f} | {j['std_sample']:.3f} |")
    lines += ['',
        '### Learning-rate and seed checks','',
        'These are actual additional CartPole experiments. Each uses 2,000 training episodes and',
        '500 evaluation episodes. Exploration of hyperparameters uses these evaluation results, so',
        'the best number is not a fresh, untouched model-selection test.','',
        '| Baseline | Learning rate | Seed | Mean reward | Sample SD |','|---|---:|---:|---:|---:|']
    trials=[]
    for file in sorted(result.glob('cartpole*/summary.json')):
        j=json.loads(file.read_text())
        if j['lr']!=.01:
            trials.append(j)
            lines.append(f"| {j['baseline']} | {j['lr']} | {j['seed']} | {j['mean']:.3f} | {j['std_sample']:.3f} |")
    lines += ['','At actor lr=0.001, matched-seed evaluation means across seeds 0, 1, and 2 were:','']
    for baseline in ['none','value']:
        subset=[j for j in trials if j['lr']==.001 and j['baseline']==baseline]
        assert sorted(j['seed'] for j in subset)==[0,1,2]
        vals=np.array([j['mean'] for j in subset])
        lines.append(f'- Baseline `{baseline}`: mean of the three policy means = {vals.mean():.3f}; between-seed sample SD = {vals.std(ddof=1):.3f}.')
    lines += ['','This between-seed SD is different from the rollout SD in each individual table row.','',
              'Reproduce the supplementary configurations by adding `--lr 0.001 --seed 0/1/2`',
              'or `--lr 0.003 --seed 0` to the CartPole commands and choosing a new output directory.','',
              '### Frozen-policy gradient variance','',
              '| Game | Held-out trajectories | Plain variance trace | Baseline variance trace | Ratio | Bootstrap 95% interval |',
              '|---|---:|---:|---:|---:|---|']
    for game,path in config['validation'].items():
        j=json.loads((ROOT/path).read_text())
        lo,hi=j['ratio_bootstrap_95_percentile']
        lines.append(f"| {game} | {j['trajectories']} | {j['plain_gradient_variance_trace']:.6g} | {j['baseline_gradient_variance_trace']:.6g} | {j['baseline_to_plain_ratio']:.6g} | [{lo:.6g}, {hi:.6g}] |")
    lines+=['','The actor is frozen, and both gradient estimators use exactly the same held-out',
            'trajectories. Ratios below 1 show empirical variance reduction at that checkpoint;',
            'ratios above 1 show increased variance. A learned baseline does not guarantee a lower',
            'variance or a better score. These samples and bootstrap intervals are exploratory.','',
            '### Pong retry versus a random policy','',
            'Each row uses paired held-out environment seeds. Intervals are approximate 95% intervals',
            'for the mean per-seed score difference; they do not measure training-seed uncertainty.','',
            '| Baseline | Mean score minus random | Approximate 95% interval | Final 100 training episodes mean entropy |',
            '|---|---:|---|---:|']
    for name in ['pong_batch_none','pong_batch_value']:
        j=summaries[name]
        ev=np.loadtxt(result/name/'evaluation.csv',skiprows=1)
        random=np.loadtxt(result/name/'random_evaluation.csv',skiprows=1)
        difference=ev-random
        se=difference.std(ddof=1)/np.sqrt(len(difference))
        training=np.genfromtxt(result/name/'training.csv',delimiter=',',names=True)
        lines.append(f"| {j['baseline']} | {difference.mean():.4f} | [{difference.mean()-1.96*se:.4f}, {difference.mean()+1.96*se:.4f}] | {np.mean(training['entropy'][-100:]):.4f} |")
    lines += ['', '![Pong training diagnostics](results/pong_batch_diagnostics.png)','',
            'Entropy near log(2)=0.693 describes continued exploration, not successful play.',
            'The exact discounted objective still downweights rewards late in a long Pong game.',
            'Lower variance and stable exploration can coexist with weak task performance.','',
            '### Separate discounted-objective diagnostic','',
            'The following uses 200 additional episodes per policy and held-out environment seeds',
            'starting at 6,000,000. It evaluates the exact full-episode gamma=0.99 objective, separate',
            'from the main 500-episode score histograms. Paired intervals describe rollout uncertainty.','',
            '| Policy | Mean discounted return | Mean game score | Discounted gain vs random (approx. 95% interval) |',
            '|---|---:|---:|---|']
    objective=json.loads((result/'pong_discounted_objective.json').read_text())
    for name,j in objective['summaries'].items():
        interval=j.get('approximate_ci95')
        gain=f"{j['discounted_improvement_over_random']:.5f} [{interval[0]:.5f}, {interval[1]:.5f}]" if interval else 'Reference'
        lines.append(f"| {name} | {j['mean_discounted_return']:.5f} | {j['mean_score']:.3f} | {gain} |")
    lines += ['','Reproduce without training: `python evaluate_pong_objective.py --episodes 200 --device cpu`.',
              'An interval including zero does not establish a clear objective gain. Even a positive',
              'discounted gain would not establish that the full game has been mastered.','',
            '### Curves and evaluation histograms','']
    for name in main_names:
        lines += [f'**{name}**','',f'![Training](results/{name}/training.png)',
                  f'![Evaluation](results/{name}/evaluation.png)','']
    lines += ['### Small environment demos','',
              'The initial CartPole smoke test used 5 training and 5 evaluation episodes. The',
              'initial Pong smoke test used 2 training and 2 evaluation episodes. Their raw results',
              'are in `results/demos/`; these are separate from all larger measured runs above.','',
              '### Report','',
              '[Complete PDF report](assignment1_report.pdf). Regenerate with `python prepare_report_figures.py`',
              'and `python build_report.py`; no new training is needed.',END]
    readme=ROOT/'README.md'
    text=readme.read_text()
    block='\n'.join(lines)
    if START in text:
        text=text[:text.index(START)]+block+text[text.index(END)+len(END):]
    else:
        text=text.replace('Measured result tables and discussion are added after the runs finish.',block)
    readme.write_text(text)
    print('Updated README from measured CSV/JSON data')


if __name__=='__main__':main()
