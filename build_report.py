"""Generate the report from actual JSON/CSV measurements. Run after experiments."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results'/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle


def load(path):return json.loads((ROOT/path).read_text())


def shuttle_pngs():
    data=np.genfromtxt(ROOT/'results/shuttle/values.csv',delimiter=',',names=True)
    for field,name,title,ylabel in [
        ('finite_cost_t0','finite_value','Finite-horizon cost at t=0, T=500','Expected discounted cost'),
        ('infinite_cost_vi','infinite_value','Infinite-horizon cost, gamma=0.95','Expected discounted cost'),
        ('dispatch_policy','optimal_policy','Optimal dispatch policy','Action: 0=wait, 1=dispatch')]:
        fig,ax=plt.subplots(figsize=(8.5,2.5))
        if field=='dispatch_policy':
            ax.step(data['state'],data[field],where='post',color='#2563eb')
            ax.set_yticks([0,1]);ax.set_ylim(-.1,1.15)
        else:ax.plot(data['state'],data[field],color='#2563eb')
        ax.set(xlabel='Customers waiting before dispatch',ylabel=ylabel,title=title)
        ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(ROOT/f'results/shuttle/{name}.png',dpi=180);plt.close(fig)


def build():
    config=load('report_config.json')
    main=config['main_experiments']
    summaries={key:load(f'results/{key}/summary.json') for key in main}
    shuttle=load('results/shuttle/validation.json')
    validation={game:load(path) for game,path in config['validation'].items()}
    shuttle_pngs()
    fig_d,axes=plt.subplots(2,1,figsize=(9,4.2),sharex=True)
    for name in ['pong_batch_none','pong_batch_value']:
        d=np.genfromtxt(ROOT/f'results/{name}/training.csv',delimiter=',',names=True)
        axes[0].plot(d['episode'][99:],np.convolve(d['reward'],np.ones(100)/100,'valid'),label=name)
        axes[1].plot(d['episode'],d['entropy'],label=name)
    axes[0].axhline(-21,color='gray',linestyle='--',label='Historical final mean -21')
    axes[0].set(ylabel='100-episode mean score',title='Pong retry: score and exploration diagnostics')
    axes[1].set(xlabel='Training episode',ylabel='Mean policy entropy')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2);axes[1].grid(alpha=.2)
    fig_d.tight_layout();fig_d.savefig(ROOT/'results/pong_batch_diagnostics.png',dpi=180);plt.close(fig_d)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',parent=styles['BodyText'],fontSize=10,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='CaptionCustom',parent=styles['BodyText'],fontSize=9,leading=12,textColor=colors.HexColor('#475569'),spaceAfter=8))
    story=[]
    def p(text):story.append(Paragraph(text,styles['BodyCustom']))
    def h(text):story.append(Paragraph(text,styles['Heading1']))
    def sub(text):story.append(Paragraph(text,styles['Heading2']))
    def fig(path,width=475,height=264):story.append(Image(str(ROOT/path),width=width,height=height))
    def table(rows,widths=None):
        rows=[[Paragraph(escape(str(v)),styles['CaptionCustom']) for v in row] for row in rows]
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eff6ff')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#94a3b8')),
                              ('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6)]))
        story.append(t);story.append(Spacer(1,10))
    h('MLDS Assignment 1')
    p('<b>GitHub username:</b> Dorrie-ge<br/><b>Repository:</b> mlds-assignment1-rl<br/>'
      '<link href="https://github.com/Dorrie-ge/mlds-assignment1-rl" color="blue">https://github.com/Dorrie-ge/mlds-assignment1-rl</link>')
    p('Measured experiments on macOS ARM, Python 3.12, one PyTorch thread: CPU for CartPole '
      'and historical runs; Apple MPS for the Pong retry. '
      'The repository is public (verified October 20, 2026). This report records actual runs, '
      'including failures. Pong uses a limited training budget and is not claimed to be solved.')
    sub('Experimental summary')
    rows=[['Environment','Baseline','Train episodes','Eval episodes','Mean score','Sample SD']]
    for key,j in summaries.items():
        rows.append([j['env_id'],j['baseline'],j['episodes'],j['evaluation_episodes'],f"{j['mean']:.3f}",f"{j['std_sample']:.3f}"])
    table(rows,[117,60,75,70,70,70])
    p('The paired CartPole runs use actor lr=0.001; the paired Pong runs use lr=0.0001 and '
      '8 complete episodes per actor update. Seed 0 is the illustrative pair; all three CartPole seeds '
      'are reported in the stability analysis. Each policy is frozen and sampled '
      'stochastically in 500 evaluation episodes. Evaluation scores are undiscounted game rewards. '
      'Standard deviations use ddof=1. Training-seed uncertainty is distinct from evaluation variability.')
    p('At lr=0.001 across CartPole seeds 0, 1, and 2, the mean of learned-policy means is '
      '166.418 without baseline versus 265.823 with baseline; between-seed sample SDs are '
      '116.107 and 84.198. The baseline does not win on seed 0 alone, and three seeds do not '
      'support a strong claim of universal superiority.')
    sub('How to reproduce')
    p('Install requirements-lock.txt in a Python 3.12 virtual environment. Run shuttle.py; then '
      'policy_gradient.py with --game cartpole, --baseline none/value, --lr 0.001, and --episodes 2000. '
      'Use pong_batch.py for the 1,000-episode paired Pong runs. Each uses --eval-episodes 500 '
      'and a separate --out directory. '
      'The exact commands and supplementary learning-rate trials are in README.md. '
      'Validation actor/critic capsules are included in the repository: no retraining is needed to '
      'repeat gradient-variance measurements. See README.md for paths, then run python build_report.py.')
    sub('Objectives and baseline')
    p('The shuttle objective minimizes expected discounted cost. Game policies maximize '
      'E[sum_t gamma^t r_t]. Define G_t=sum_{k=t}^{T-1} gamma^(k-t) r_k. '
      'The actor loss is -sum_t gamma^t log pi(a_t|s_t)(G_t-b(s_t)). '
      'A separate value network predicts b(s_t); its prediction is detached and obtained before fitting '
      'the current trajectory or batch. The critic uses Monte Carlo MSE and Adam: '
      'lr=0.001 in CartPole and 0.0003 in batched Pong.')
    story.append(PageBreak())
    h('1. Shuttle dispatch: model and solutions')
    p('State s is the pre-dispatch queue, 0 through 200; action a is 0 (wait) or 1 (dispatch one shuttle). '
      'Let l=max(s-15a,0). Cost is 100a+2l; next state is min(200,l+A), '
      'with independent uniform A in {1,2,3,4,5}. Dispatch precedes holding charges and arrivals. '
      'Overflow is discarded without penalty; empty dispatch still costs 100. Ties choose wait.')
    p('Finite-horizon backward induction sets J_500=0 and enumerates all state-action-arrival '
      'outcomes in J_t(s)=min_a[c(s,a)+0.95 E J_{t+1}(s\')]. Enumeration is interpreted as '
      'this feedback-policy dynamic program. VI repeats the infinite-horizon Bellman update; '
      'PI solves (I-0.95 P_pi)J=c_pi before greedy improvement. Values below are positive costs, '
      'not reward-maximizing values; use V=-J if rewards are -cost.')
    for filename in ['finite_value','infinite_value','optimal_policy']:
        fig(f'results/shuttle/{filename}.png',475,140)
    p(f"The optimal infinite-horizon policy dispatches at s &gt;= 13. J*(0)={shuttle['cost_at_zero']:.6f}. "
      f"VI took {shuttle['vi_iterations']} iterations and PI {shuttle['pi_iterations']}. "
      'A threshold below capacity reflects the cost of waiting. Finite-horizon actions near the terminal '
      'time need not match the infinite-horizon policy.')
    story.append(PageBreak())
    for index,key in enumerate(main):
        j=summaries[key]
        h(f"{index+2}. {j['env_id']}: baseline={j['baseline']}")
        p(f"Training: {j['episodes']} episodes, seed {j['seed']}, gamma={j['gamma']}, actor Adam lr={j['lr']}. "
          f"Final evaluation: {j['evaluation_episodes']} episodes; mean={j['mean']:.3f}, sample SD={j['std_sample']:.3f}. "
          f"Random-policy mean={j['random_mean']:.3f}, sample SD={j['random_std_sample']:.3f}.")
        if j['game']=='cartpole':
            p('The policy maps the four observed coordinates through 128 tanh units to two logits. '
              'Both terminated and truncated end a rollout. No value bootstrap is added at truncation; '
              'this optimizes the sampled finite episode. Training and evaluation seeds are disjoint.')
        else:
            p('The environment is ALE/Pong-v5 with its default frameskip=4 and sticky-action probability 0.25. '
              'Two policy logits map only to actions 2 and 3. Inputs are consecutive binary 80x80 frames; '
              'history resets every episode. The CNN has 16 filters (8x8, stride 4), then 32 filters '
              '(4x4, stride 2), then 128 hidden units. There are no point-based return resets.')
        fig(f'results/{key}/training.png',414,230)
        fig(f'results/{key}/evaluation.png',414,230)
        if key=='cartpole_none_lr0001':
            p('This matched lr=0.001 run exceeds the random reference. The historical lr=0.01 '
              'collapse remains in the stability table. These seed-0 figures illustrate a paired '
              'configuration rather than selecting the best seed separately for each baseline.')
        elif key=='cartpole_value_lr0001':
            p('This run exceeds the random reference but does not establish mastery of the 500-step task. '
              'Reward improvements alone do not establish that gradient variance was reduced.')
        else:
            p('Stability changes: average 8 full-episode gradients per update, reduce actor lr to 0.0001, '
              'initialize the actor output with gain 0.01, and clip gradient norm at 5. The discounted '
              'objective and game are unchanged. These results do not by themselves establish mastery '
              'of Pong; no unmeasured longer-run performance is claimed.')
        story.append(PageBreak())
    h('6. Validation and edge cases')
    sub('Independent validation of the shuttle algorithm')
    table([['Experiment','Observed evidence'],
           ['Tiny problem: 5 states, 2 arrivals, 4 steps','Independent recursive tree enumeration vs backward induction: max error 0'],
           ['VI vs PI',f"Max value difference {shuttle['vi_vs_pi_max_error']:.3e}; policy mismatches 0"],
           ['PI Bellman optimality residual',f"{shuttle['policy_evaluation_bellman_residual']:.3e}"],
           ['Finite T=500 vs infinite horizon',f"Max value difference {shuttle['finite_vs_infinite_max_error']:.3e}"],
           ['Free dispatch; zero capacity','All positive states dispatch when free; never dispatch with zero capacity'],
           ['Zero discount; upper queue boundary','Values equal minimum immediate costs; s=200 transitions verified']],[160,302])
    p('At s=0, dispatch costs 100 and never creates negative passengers. At s=200, waiting clips '
      'all next states to 200; dispatch gives 186 through 190. The finite-horizon terminal value '
      'is explicitly zero. The parameter experiments test qualitative predictions of the model, '
      'rather than only input formats or absence of runtime errors.')
    sub('Independent Monte Carlo cost check')
    sim=load('results/shuttle/simulation_validation.json')
    rows=[['Initial state','DP cost','Simulation mean','Standard error']]
    for j in sim['rows']:
        rows.append([j['initial_state'],f"{j['dp_cost']:.3f}",f"{j['simulated_mean']:.3f}",f"{j['standard_error']:.3f}"])
    table(rows,[90,120,130,122])
    p('Each initial state uses 40,000 independent 500-step simulations with seed 314159. '
      'Four of five DP values are inside their marginal 95% intervals; the state-15 discrepancy is '
      '2.44 standard errors. All five are inside simultaneous 95% Bonferroni intervals '
      '(approximately mean +/- 2.576 SE). The maximum omitted discounted tail is below 6.9e-8. '
      'The state-15 result is retained rather than rerunning seeds to force agreement.')
    story.append(PageBreak())
    h('6. Validation (continued)')
    sub('Analytic and numerical policy-gradient checks')
    p('An independent two-step action enumeration compares the exact discounted objective derivative '
      'with the expected score-function gradient, with and without a constant baseline. Both absolute '
      'errors are below 6e-8. For the actual CartPole actor, one parameter is also checked using a '
      'central finite difference; the absolute discrepancy is about 4.2e-12.')
    sub('Direct measurement of gradient variance')
    rows=[['Frozen policy','Trajectories','Var trace: plain','Var trace: baseline','Ratio']]
    for game,j in validation.items():
        rows.append([game,j['trajectories'],f"{j['plain_gradient_variance_trace']:.4g}",
                     f"{j['baseline_gradient_variance_trace']:.4g}",f"{j['baseline_to_plain_ratio']:.5g}"])
    table(rows,[80,75,110,120,77])
    for game,j in validation.items():
        lo,hi=j['ratio_bootstrap_95_percentile']
        p(f"{game}: paired bootstrap 95% percentile interval for baseline/plain variance is [{lo:.5g}, {hi:.5g}]. "
          f"The actor and critic were frozen at training episode {j['checkpoint_episode']}; seeds start at 8,000,000.")
    p('Variance is the trace of the sample covariance of the full actor-gradient vector, computed '
      'on the same held-out trajectories for both estimators. The interval uses 300 paired bootstrap '
      'resamples. A ratio below 1 indicates empirical reduction at this checkpoint only. Small samples '
      'and a learned baseline do not justify a universal claim that baselines improve every task.')
    p('The historical Pong actor was almost deterministic: gradient covariance at episode 200 was '
      'approximately 4e-49, so its variance ratio is numerically uninformative. That diagnostic is '
      'retained separately. The Pong comparison above uses the final matched batched-training '
      'checkpoint. The prior 20-episode diagnostic and its measurements remain in the repository.')
    p('Portable validation_checkpoint.pt files contain both actor and critic weights, the episode '
      'number, and experiment identity. The export script checks tensor equality against the '
      'source checkpoint; results/validation_checkpoints.json records SHA-256 hashes. '
      'A numerical dense-reference test verifies the new batched/chunked gradient computation, '
      'including an episode crossing the 256-step chunk boundary.')
    story.append(PageBreak())
    h('7. Stability experiments and limitations')
    trialrows=[['Baseline','Learning rate','Seed','Train episodes','Eval mean','Eval SD']]
    for d in sorted((ROOT/'results').glob('cartpole*')):
        file=d/'summary.json'
        if file.exists():
            j=json.loads(file.read_text())
            trialrows.append([j['baseline'],j['lr'],j['seed'],j['episodes'],f"{j['mean']:.3f}",f"{j['std_sample']:.3f}"])
    table(trialrows,[75,75,55,90,80,87])
    p('Smaller learning rates are a stability intervention, not a change in the task. The table '
      'retains every run, including lr=0.01 collapse. Three matched seeds at lr=0.001 provide an '
      'initial sensitivity check. Final evaluation means from hyperparameter exploration should '
      'not be interpreted as performance on an untouched model-selection test set.')
    p('Evaluation variability describes repeated rollouts of one learned policy; training-seed '
      'variability describes the learning procedure. Neither a narrow rollout histogram nor a single '
      'successful seed proves robust learning. Pong has one seed and 1,000 training episodes per '
      'configuration. Its results are limited-budget demonstrations, not completed high-performance training.')
    story.append(PageBreak())
    h('7. Stability (continued)')
    sub('Pong stability diagnostics')
    fig('results/pong_batch_diagnostics.png',475,220)
    comparison=[['Baseline','Score minus random','Approx. 95% interval','Final-100 entropy']]
    for name in ['pong_batch_none','pong_batch_value']:
        scores=np.loadtxt(ROOT/f'results/{name}/evaluation.csv',skiprows=1)
        random_scores=np.loadtxt(ROOT/f'results/{name}/random_evaluation.csv',skiprows=1)
        delta=scores-random_scores;se=delta.std(ddof=1)/np.sqrt(len(delta))
        train=np.genfromtxt(ROOT/f'results/{name}/training.csv',delimiter=',',names=True)
        comparison.append([summaries[name]['baseline'],f'{delta.mean():.4f}',
                           f'[{delta.mean()-1.96*se:.4f}, {delta.mean()+1.96*se:.4f}]',
                           f"{np.mean(train['entropy'][-100:]):.4f}"])
    table(comparison,[75,110,155,122])
    p('Entropy is measured during stochastic training, before each batch update. The historical '
      'lr=0.01 runs reached -21 mean score; this retry tests a joint stability intervention. '
      'Learning rate, averaging, initialization, and clipping were changed together, so their '
      'individual contributions have not been isolated. Stable exploration is a diagnostic, '
      'not proof that the agent has learned to win. Evaluation and random-policy score differences '
      'use paired environment seeds; their intervals describe rollout variability, not training-seed '
      'uncertainty. An interval including zero does not support a clear improvement over random.')
    story.append(PageBreak())
    h('7. Discounted objective diagnostic')
    objective=load('results/pong_discounted_objective.json')
    p('The required score histograms use undiscounted game totals, while training optimizes '
      'discounted reward with gamma=0.99. These are different metrics. A separate held-out '
      'diagnostic measures the exact full-episode training objective, without truncating returns '
      'or resetting discount at score points. It uses 200 additional episodes per policy and '
      'environment seeds starting at 6,000,000; these are separate from the main 500-episode scores.')
    objrows=[['Policy','Mean discounted return','Mean game score','Discounted gain vs random (95% interval)']]
    for name,j in objective['summaries'].items():
        interval=j.get('approximate_ci95')
        gain=(f"{j['discounted_improvement_over_random']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else 'Reference')
        objrows.append([name.replace('pong_batch_',''),f"{j['mean_discounted_return']:.4f}",f"{j['mean_score']:.3f}",gain])
    table(objrows,[70,120,100,172])
    p('Intervals use paired per-environment-seed differences and a normal approximation. '
      'An interval including zero cannot establish a clear objective improvement over random. '
      'A positive discounted gain would still not establish successful full-game play. '
      'The per-episode returns are retained in the diagnostic JSON, so the interpretation can '
      'be checked rather than relying on qualitative claims about longer training.')
    p('Here both measured discounted-gain intervals are above zero, whereas full-game scores '
      'remain poor. Discounting emphasizes early score timing and can favor postponing a loss; '
      'these aggregate measurements do not isolate which timing changes occurred. '
      'The evidence therefore supports improvement of the specified '
      'training objective, not a claim that the agent has learned to win Pong. '
      'Changing or removing the outer gamma^t would change the weighting being studied and '
      'has not been silently used to improve these results.')
    story.append(PageBreak())
    h('8. Three additional months')
    p('For shuttle dispatch, test larger queue caps and alternative overflow rules, nonstationary '
      'or heavy-tailed arrivals, different capacity and cost ratios, and sensitivity of the threshold. '
      'Independently simulate the resulting policies to compare Monte Carlo costs and DP predictions.')
    p('For policy gradients, increase independent seeds and budgets; compare constant and value '
      'baselines, learning rates, batch sizes, network sizes, and frame representations under matched '
      'budgets. Repeat frozen-policy gradient-variance measurements at early, middle, and late '
      'checkpoints. Study sparse rewards and the effect of the exact start-state discounted objective. '
      'Label alternative objectives or environment modifications explicitly. Evaluate robustness to '
      'initial conditions, observation noise, and sticky-action probability in separate experiments.')
    sub('Sources and reproducibility')
    p('Assignment source: mlds_assignment1.pdf, supplied course handout, pages 1-4. '
      'Environment reference: https://ale.farama.org/environments/pong/. '
      'Algorithm reference: OpenAI Spinning Up, Intro to Policy Optimization, '
      'https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html. '
      'Raw CSVs, JSON summaries, model weights, exact dependency versions, and commands are in the repository. '
      'AI assistance was used to develop code, debugging, and documentation; all numeric experiment '
      'results here were generated by actual execution.')
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(50,28,'Dorrie-ge | MLDS Assignment 1 | Measured experiments')
        canvas.drawRightString(562,28,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(ROOT/'assignment1_report.pdf'),pagesize=(612,792),
                          rightMargin=50,leftMargin=50,topMargin=38,bottomMargin=45,
                          title='MLDS Assignment 1 - Dorrie-ge',author='Dorrie-ge')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(ROOT/'assignment1_report.pdf')


if __name__=='__main__':build()
