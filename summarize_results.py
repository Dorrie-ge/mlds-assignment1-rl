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
    main_names=['cartpole_none','cartpole_value','pong_none','pong_value']
    summaries={name:json.loads((result/name/'summary.json').read_text()) for name in main_names}
    lines=[START,'## Measured results','',
           'Main experiments, seed 0 and actor learning rate 0.01. All evaluation budgets below are 500 episodes.','',
           '| Environment | Baseline | Training episodes | Mean reward | Sample SD | Random-policy mean |',
           '|---|---|---:|---:|---:|---:|']
    for name,j in summaries.items():
        lines.append(f"| {j['env_id']} | {j['baseline']} | {j['episodes']} | {j['mean']:.3f} | {j['std_sample']:.3f} | {j['random_mean']:.3f} |")
        rewards=np.loadtxt(result/name/'evaluation.csv',skiprows=1)
        assert len(rewards)==500
        assert abs(rewards.mean()-j['mean'])<1e-8
        assert abs(rewards.std(ddof=1)-j['std_sample'])<1e-8
    lines += ['',
        'The default-lr CartPole run without a baseline collapsed. The value baseline improved that',
        'particular run. Pong did not learn successful play within 200 training episodes; its weak',
        'results are retained. No performance for 20,000 or more training episodes is claimed.','',
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
    for game in ['cartpole','pong']:
        j=json.loads((result/f'validation_{game}.json').read_text())
        lo,hi=j['ratio_bootstrap_95_percentile']
        lines.append(f"| {game} | {j['trajectories']} | {j['plain_gradient_variance_trace']:.6g} | {j['baseline_gradient_variance_trace']:.6g} | {j['baseline_to_plain_ratio']:.6g} | [{lo:.6g}, {hi:.6g}] |")
    lines+=['','The actor is frozen, and both gradient estimators use exactly the same held-out',
            'trajectories. Ratios below 1 show empirical variance reduction at that checkpoint;',
            'ratios above 1 show increased variance. A learned baseline does not guarantee a lower',
            'variance or a better score. These samples and bootstrap intervals are exploratory.','',
            '### Curves and evaluation histograms','']
    for name in main_names:
        lines += [f'**{name}**','',f'![Training](results/{name}/training.png)',
                  f'![Evaluation](results/{name}/evaluation.png)','']
    lines += ['### Small environment demos','',
              'The initial CartPole smoke test used 5 training and 5 evaluation episodes. The',
              'initial Pong smoke test used 2 training and 2 evaluation episodes. Their raw results',
              'are in `results/demos/`; these are separate from all larger measured runs above.','',
              '### Report','',
              '[PDF report](assignment1_report.pdf). Regenerate with `python summarize_results.py`',
              'and `python build_report.py` after the measurements and variance experiments finish.',END]
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
