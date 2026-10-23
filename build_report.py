"""Build the complete assignment report from audited, actually measured results.

No training is launched. Requires the independent seed confirmation to finish.
Run prepare_report_figures.py first, then this script with ReportLab and NumPy.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import hashlib,json,re,textwrap
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,KeepTogether
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parent
BLUE=colors.HexColor('#173c62');MUTED=colors.HexColor('#526477');LIGHT=colors.HexColor('#edf3f8')

def load(path):return json.loads((ROOT/path).read_text())
def summary(name):return load(f'results/{name}/summary.json')
def score(name):
    s=summary(name);return f"{s['mean']:.3f} +/- {s['std_sample']:.3f}"
def fnum(x):return f'{x:.3f}'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    config=load('report_config.json');main=config['main_experiments']
    seed=load('results/pong_seed_validation.json')
    assert len(seed['runs'])==3 and len(seed['confirmation_seeds'])==2
    audit=load('results/audit.json');assert audit['passed']
    evidence=[]
    for row in audit['experiments']:
        name=row['experiment'];s=summary(name);directory=ROOT/f'results/{name}'
        ev=np.loadtxt(directory/'evaluation.csv',skiprows=1)
        train=np.atleast_1d(np.genfromtxt(directory/'training.csv',delimiter=',',names=True))
        assert len(ev)==s['evaluation_episodes'] and len(train)==s['episodes']
        if name!='pong_value_early_diagnostic':assert len(ev)==500
        assert abs(ev.mean()-s['mean'])<1e-10 and abs(ev.std(ddof=1)-s['std_sample'])<1e-10
        for image in ['training.png','evaluation.png']:assert (directory/image).exists()
        evidence.append(dict(experiment=name,train_episodes=len(train),evaluation_episodes=len(ev),mean=float(ev.mean()),sample_sd=float(ev.std(ddof=1)),training_csv_sha256=digest(directory/'training.csv'),evaluation_csv_sha256=digest(directory/'evaluation.csv'),summary_sha256=digest(directory/'summary.json')))
    supporting=['results/shuttle/values.csv','results/shuttle/validation.json','results/shuttle/simulation_validation.json',*config['validation'].values(),'results/pong_discounted_objective.json','results/pong_control_diagnostic.json','results/cartpole_control_diagnostic.json','results/pong_endpoint_activations.json',*config['supplementary_results']]
    (ROOT/'results/report_evidence.json').write_text(json.dumps(dict(source_assignment_sha256=config['source_assignment_sha256'],source_assignment_pages=4,report_date='2026-10-23',audited_experiments=len(evidence),core_experiments=main,experiments=evidence,supporting_input_sha256={path:digest(ROOT/path) for path in supporting}),indent=2))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=27,leading=31,textColor=BLUE,spaceAfter=14))
    styles.add(ParagraphStyle(name='ReportH1',fontName='Helvetica-Bold',fontSize=18,leading=22,textColor=BLUE,spaceAfter=12))
    styles.add(ParagraphStyle(name='ReportH2',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=BLUE,spaceBefore=7,spaceAfter=7))
    styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=10,leading=13.7,spaceAfter=8))
    styles.add(ParagraphStyle(name='ReportSmall',fontName='Helvetica',fontSize=8.6,leading=11.5,textColor=MUTED,spaceAfter=6))
    styles.add(ParagraphStyle(name='ReportCell',fontName='Helvetica',fontSize=8.5,leading=11))
    styles.add(ParagraphStyle(name='ReportCode',fontName='Courier',fontSize=7.5,leading=10,spaceAfter=6,backColor=LIGHT,borderPadding=7))
    story=[];sections=[];figure_number=0
    def p(text,small=False):story.append(Paragraph(text,styles['ReportSmall' if small else 'ReportBody']))
    def h(text):
        sections.append(text);story.append(Paragraph(text,styles['ReportH1']))
    def sub(text):story.append(Paragraph(text,styles['ReportH2']))
    def page():story.append(PageBreak())
    def code(text):
        lines=[]
        for line in text.splitlines():
            if line.startswith(('python ','python3 ','source ')):
                chunks=textwrap.wrap(line,width=105,break_long_words=False,break_on_hyphens=False,subsequent_indent='  ')
                lines.append(' \\\n'.join(chunks))
            else:lines.append(line)
        story.append(Paragraph(escape('\n'.join(lines)).replace('\n','<br/>'),styles['ReportCode']))
    def table(rows,widths=None,compact=False):
        values=[[Paragraph(escape(str(v)),styles['ReportCell']) for v in row] for row in rows]
        t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        padding=3 if compact else 6
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.65,colors.HexColor('#a2b4c7')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding)]))
        story.extend([t,Spacer(1,9)])
    def figure(path,caption,width=500,height=215):
        nonlocal figure_number
        figure_number+=1
        caption=f'Figure {figure_number}. '+re.sub(r'^Figure [0-9]+\.\s*','',caption)
        path=ROOT/path;w,hp=ImageReader(str(path)).getSize();scale=min(width/w,height/hp)
        image=Image(str(path),width=w*scale,height=hp*scale);image.hAlign='CENTER'
        story.append(KeepTogether([image,Spacer(1,4),Paragraph(caption,styles['ReportSmall'])]))
    def required_page(section,name):
        s=summary(name);h(section)
        p(f"<b>Configuration:</b> {s['env_id']}, baseline={s['baseline']}, seed {s['seed']}, gamma={s['gamma']}, actor Adam lr={s['lr']}. Training: {s['episodes']:,} episodes. Frozen stochastic-policy evaluation: 500 episodes.")
        table([['Mean score','Sample SD','Random-policy mean'],[f"{s['mean']:.3f}",f"{s['std_sample']:.3f}",f"{s['random_mean']:.3f}"]],[174,174,172])
        figure(f'results/{name}/training.png',f"Training reward and trailing 100-episode mean. Source: results/{name}/training.csv.",height=217)
        figure(f'results/{name}/evaluation.png',f"Histogram of the 500 frozen-policy game scores; random-policy reference is displayed separately. Standard deviations use ddof=1.",height=217)
        return s

    story.append(Paragraph('MLDS Assignment 1',styles['ReportTitle']))
    p('<b>From initial policy-gradient failures to controlled improvements</b><br/>Complete report | 23 October 2026')
    p('<b>GitHub username:</b> Dorrie-ge<br/><b>Repository:</b> mlds-assignment1-rl<br/><link href="https://github.com/Dorrie-ge/mlds-assignment1-rl" color="#2563eb">https://github.com/Dorrie-ge/mlds-assignment1-rl</link>')
    sub('Main findings')
    p('The shuttle MDP has a dispatch threshold of 13 passengers; value iteration and policy iteration agree. CartPole learning improves after reducing the original Adam learning rate, with considerable variation between training seeds. Pong initially remains near a -21 game score. Extending the same configuration, changing actor time weights and changing image representation produce only modest gains. A matched actor-learning-rate comparison then produces a much larger gain at lr=0.001.')
    table([['Core experiment (strict discounted objective)','Train','500-episode mean +/- SD'],*[[f"{summary(n)['env_id']}, b={summary(n)['baseline']}",summary(n)['episodes'],score(n)] for n in main]],[260,60,200])
    p(f"<b>Improved Pong extension:</b> lr=0.001 reaches {score('pong_motion_lr001_value')} on selection seed 0. The two independent confirmation seeds average {seed['independent_confirmation_mean']:.3f}. Across the three policy means, the average is {seed['three_seed_mean']:.3f} and the sample SD is {seed['sample_sd_between_three_seed_means']:.3f}. Scores remain negative on average; this is improved learning, not stable mastery.")
    p('<b>Objective distinction.</b> Core experiments use the exact start-state discounted objective requested in the handout. Later Pong extensions retain gamma=0.99 inside reward-to-go but remove the outer gamma^t actor weight. This is a different update weighting and is reported explicitly as a supplementary method, not silently substituted for the core objective.')
    p('Reading order: sections 1-3 define the model and algorithms; sections 4-7 contain all four required plot pairs; sections 8-13 reconstruct the improvement experiments; sections 14-17 cover substantive validation, edge cases, execution instructions and future work. Appendix A inventories every completed experiment.',small=True)
    page()
    h('1. Shuttle MDP and solution methods')
    p('The decision state s is the number of passengers waiting before dispatch: s in {0,...,200}. The action a is 0 (wait) or 1 (dispatch one shuttle). Capacity K=15, dispatch cost c_f=100, holding cost c_h=2, discount gamma=0.95 and independent arrivals A uniformly distributed on {1,2,3,4,5} follow the assignment.')
    sub('Timing and boundary assumptions')
    code("left(s,a) = max(s - 15*a, 0)\nc(s,a) = 100*a + 2*left(s,a)\ns_next = min(200, left(s,a) + A)")
    p('Dispatch happens before holding cost and arrivals. Overflow above 200 is discarded without an extra penalty. Empty dispatch is permitted but still costs 100. Ties choose wait. These assumptions define a finite-state controlled Markov chain; no service or arrival history beyond the queue is needed.')
    sub('Finite horizon: enumeration with T=500')
    code("J_500(s) = 0\nJ_t(s) = min_a {c(s,a) + 0.95*(1/5)*sum_A J_(t+1)(s_next)}")
    p('Backward induction enumerates all 201 states, two actions and five arrival outcomes at each time. This produces an optimal feedback policy, including different decisions near the terminal time. The reported time-zero value is J_0(s). Direct enumeration of 2^500 open-loop action strings would be intractable and would not represent every feedback policy; the short-horizon tree validation in section 14 independently checks the Bellman enumeration interpretation.')
    sub('Infinite horizon: VI and PI')
    p('Value iteration starts from zero and applies the discounted Bellman minimum operator until the maximum update residual is below 1e-10. Policy iteration exactly evaluates the current policy with (I-0.95 P_pi)J=c_pi, then greedily improves it until no action changes. Both methods minimize positive costs; a reward formulation would use r=-c and V=-J.')
    shuttle=load('results/shuttle/validation.json')
    table([['Outcome','Measured value'],['VI iterations',shuttle['vi_iterations']],['PI iterations',shuttle['pi_iterations']],['Optimal cost J*(0)',f"{shuttle['cost_at_zero']:.9f}"],['Infinite-horizon policy','Wait at 0-12; dispatch at 13-200']],[260,260])
    page()
    h('2. Required shuttle plots')
    figure('results/shuttle/finite_value.png','Figure 1. Enumeration / backward induction: optimal positive discounted cost at time 0, horizon T=500.',height=158)
    figure('results/shuttle/infinite_value.png','Figure 2. Value iteration: optimal infinite-horizon positive discounted cost.',height=158)
    figure('results/shuttle/optimal_policy.png','Figure 3. Policy iteration: action 0=wait, action 1=dispatch; threshold 13. All queue states 0-200 are shown.',height=158)
    p('The threshold can lie below capacity: avoiding another holding period may justify dispatching a partly filled shuttle. At large queues the value curve has service-cycle structure, while the capped queue preserves a bounded state space. The finite T=500 and infinite-horizon values differ by less than 3e-9, but their late-terminal policies are not assumed identical. Raw values are stored in results/shuttle/values.csv.')
    page()
    h('3. Policy gradient and the value baseline')
    sub('Core objective and estimator')
    code("J(theta) = E[sum_t gamma^t * r_t]\nG_t = sum_(k=t)^(T-1) gamma^(k-t) * r_k\nL_actor = -(1/B) * sum_episodes sum_t gamma^t * log pi_theta(a_t|s_t) * (G_t - b(s_t))")
    p('Categorical actions are sampled from neural-network logits. PyTorch autograd computes the loss gradient and Adam updates the actor. There is no PPO clipping, reward clipping, return standardization, value bootstrap, or return reset at Pong score points. CartPole uses gamma=0.95; Pong uses gamma=0.99. Evaluation always reports the undiscounted whole-game reward sum, a different metric from the discounted training objective.')
    sub('Part 2: nonconstant, learned state-value baseline')
    p('A separate network b_phi(s) predicts Monte Carlo reward-to-go. Its output is computed before fitting the current trajectory or batch and detached from the actor graph. The critic uses squared prediction error and its own Adam optimizer. The actor therefore cannot improve its loss by differentiating through the baseline. In CartPole the critic is fitted on trajectory chunks; in batched Pong one averaged critic update follows each actor update, with each trajectory normalized by its length.')
    code("E_a[b(s) * grad log pi(a|s)] = b(s) * grad sum_a pi(a|s) = 0")
    p('This action-independent baseline leaves the expected score-function estimator unchanged for the chosen weighting. It can reduce variance, but neither a value-function fit nor a lower variance guarantees higher game scores. Direct frozen-policy measurements appear in section 15.')
    table([['Setting','CartPole core','Pong core'],['Observation / actor','4 coordinates; 128 tanh units; 2 logits','2 binary 80x80 frames; CNN; 2 logits'],['CNN (Pong only)','Not applicable','Conv 16: 8x8/stride4; Conv 32: 4x4/stride2; FC128/ReLU'],['Actor / critic lr','0.001 / 0.001','0.0001 / 0.0003'],['Episodes / batch','2,000 / 1','1,000 / 8'],['Optimizer / hardware','Adam; CPU, 1 thread','Adam; Apple MPS, 1 CPU thread']],[130,180,210])
    p('Environment IDs are CartPole-v1 and ALE/Pong-v5, the Gymnasium/ALE registration of the requested Pong-v5. Pong retains the default frameskip=4 and sticky-action probability=0.25. The policy maps its two logits only to actions 2 and 3. It never chooses FIRE or NOOP. Episode history resets on termination or truncation. [R2-R4]',small=True)
    page()
    required_page('4. Part 1: CartPole without baseline','cartpole_none_lr0001')
    p('Reducing actor lr from 0.01 to 0.001 changes the seed-0 evaluation mean from 9.370 to 300.240. The policy is substantially better than random but has not consistently achieved the 500-step limit. It remains stochastic during evaluation.')
    page()
    required_page('5. Part 2: CartPole with value baseline','cartpole_value_lr0001')
    p('This matched seed-0 run scores 220.218, below the no-baseline seed-0 result. Across all three matched seeds the baseline has a higher average, illustrating why one favorable run cannot establish a general baseline benefit. Section 15 measures gradient variance directly.')
    page()
    required_page('6. Part 1: Pong without baseline','pong_batch_none')
    p('This strict-objective retry averages eight complete episodes per update, uses actor-head orthogonal initialization with gain 0.01, and clips gradient norm at 5. It still scores near -21. The failure is retained as measured evidence, rather than replacing its histogram with an inferred larger-run outcome.')
    page()
    required_page('7. Part 2: Pong with value baseline','pong_batch_value')
    p('The independent CNN critic is detached in the actor update and fitted after it. This run is slightly worse than its matched no-baseline counterpart. Later variance tests show reduced estimator variance at this checkpoint, but the score evidence does not establish that the baseline alone improves play.')
    page()
    h('8. Starting point and CartPole improvements')
    p('The first full experiments used the handout Adam actor learning rate 0.01. CartPole trained for 2,000 episodes per baseline and Pong for 200 episodes per baseline. The initial implementation used single-episode REINFORCE and the exact outer discounted weight. Every final policy received 500 evaluation rollouts. Small environment smoke tests were separate and were not treated as assignment results.')
    table([['Initial experiment','Actor lr','Train','500-episode mean +/- SD'],*[[n,summary(n)['lr'],summary(n)['episodes'],score(n)] for n in ['cartpole_none','cartpole_value','pong_none','pong_value']]],[205,65,55,195])
    p('CartPole without a baseline collapsed below its random reference; Pong reached -21 with zero evaluation variation. These observations motivated parameter changes. For CartPole, lr=0.003 was explored without a baseline and scored 171.432. A matched lr=0.001 comparison was then run at seeds 0, 1 and 2, while preserving the task, architecture and objective.')
    rows=[['Training seed','No baseline mean +/- SD','Value baseline mean +/- SD']]
    for k in [0,1,2]:
        suffix='' if k==0 else f'_seed{k}';rows.append([k,score('cartpole_none_lr0001'+suffix),score('cartpole_value_lr0001'+suffix)])
    table(rows,[100,210,210])
    for baseline in ['none','value']:
        means=np.array([summary(f'cartpole_{baseline}_lr0001'+('' if k==0 else f'_seed{k}'))['mean'] for k in [0,1,2]])
        p(f"{baseline}: mean of the three policy means = {means.mean():.3f}; between-training-seed sample SD = {means.std(ddof=1):.3f}.")
    p('This supports choosing a smaller CartPole learning rate for the submitted figures, not a claim that one value-baseline run always outperforms one plain run. The evaluation rollouts used during exploration are not an untouched hyperparameter-selection test set.')
    sub('CartPole task-level sanity check')
    cartdiag=load('results/cartpole_control_diagnostic.json')
    table([['Deterministic diagnostic (20 episodes each)','Mean reward','Episode ending'],['Always push left',f"{cartdiag['mean_scores']['always_left']:.2f}",'Physical termination'],['Angle + 0.15*angular-velocity feedback',f"{cartdiag['mean_scores']['angle_feedback']:.2f}",'All 20 reach the 500-step time limit']],[295,80,145])
    p('The predictions are rapid failure under a fixed push and sustained balance under state feedback. Both are observed on separate seeds starting at 9,500,000; reward equals episode length. First-10-step traces retain states, actions and ending flags. This validates state/action conventions, the reward scale and time-limit handling. The controller is hand-coded and is not substituted for a learned-policy score.',small=True)
    page()
    h('9. Pong: stability retry and larger budgets')
    p('After the original lr=0.01 failure, the Pong retry jointly lowered actor lr to 0.0001, averaged 8 full trajectories per update, initialized the actor output near an even action distribution and clipped gradients. It retained the CNN, stacked frames, gamma=0.99 and exact outer gamma^t. Because several stabilizers changed together, the initial retry cannot isolate their individual effects.')
    p('The next experiment changed only total training budget. Each 1,000-episode actor/critic, Adam state and CPU sampling RNG was copied to an independent directory and resumed to 3,000, then 5,000 episodes. Source CSVs remain exact prefixes; original results were not overwritten. Each endpoint received 500 evaluation episodes.')
    rows=[['Total episodes','Actor updates','No baseline mean +/- SD','Value baseline mean +/- SD']]
    for n,prefix in [(1000,'pong_batch'),(3000,'pong_budget3000'),(5000,'pong_budget5000')]:rows.append([n,n//8,score(prefix+'_none'),score(prefix+'_value')])
    table(rows,[90,90,170,170])
    figure('results/report_budget_comparison.png','Figure 12. Measured evaluation means at three budgets; each point represents a separately evaluated endpoint, not an inferred score.',height=225)
    p('At 3,000 episodes the no-baseline result deteriorated; the baseline improved only slightly. At 5,000 both remain poor. Thus this configuration did not become successful merely by extending training. The experiment does not prove that more data can never help; it demonstrates that budget alone was not an effective fix over the tested range. Subsequent experiments hold each fresh-run budget at 3,000 episodes.')
    page()
    h('10. Pong: the actor time-weighting experiment')
    p('The exact start-state discounted objective gives late actions very little weight: 0.99^500 is approximately 0.00657 and 0.99^1000 approximately 0.0000432. This is mathematically appropriate for that objective, but it can conflict with the desire to improve an entire long-game score. It motivated an explicitly labeled alternative weighting, not a claim that the original estimator was a coding error.')
    code("Core:     loss = -sum_t gamma^t * log pi(a_t|s_t) * (G_t - b(s_t))\nExtension: loss = -sum_t           log pi(a_t|s_t) * (G_t - b(s_t))")
    p('The extension keeps full-episode gamma=0.99 reward-to-go and removes only the outer time factor. Both branches start from the same 3,000-episode checkpoint and train to 5,000; uniform weighting starts at episode 3,001. Architecture, learning rates, batch, clipping and seed remain fixed. This is a warm-start experiment with inherited optimizer state.')
    weights=load('results/pong_time_weighting_comparison.json')
    rows=[['Baseline','Original weight mean','Uniform weight mean','Gain (approx. paired 95% CI)']]
    for baseline in ['none','value']:
        a=next(x for x in weights['runs'] if x['kind']=='control' and x['baseline']==baseline);b=next(x for x in weights['runs'] if x['kind']=='uniform' and x['baseline']==baseline);d=next(x for x in weights['paired_differences'] if x['baseline']==baseline);lo,hi=d['approximate_paired_ci95'];rows.append([baseline,fnum(a['mean']),fnum(b['mean']),f"{d['uniform_minus_control']:.3f} [{lo:.3f}, {hi:.3f}]"])
    table(rows,[75,110,110,225])
    figure('results/pong_time_weighting_comparison.png','Figure 13. Original and uniform-weight training branches; the vertical line marks the shared 3,000-episode fork.',height=205)
    p('The small gains are distinguishable from rollout variability for these fixed policies, but neither branch learns effective play. The intervals do not describe variation across training seeds. This practical surrogate is retained for the later exploratory ablations; the strict-objective core results remain separately reported.')
    page()
    h('11. Pong: motion representation at fixed budget')
    p('The stacked-frame CNN must infer motion from two binary images. The alternative explicitly encodes appearing and disappearing pixels in separate binary channels. Subtracting these two channels reconstructs current_frame-previous_frame. Two channels preserve the CNN shape and preserve the sign through packed binary trajectory storage. Directly packing signed -1/0/1 differences would lose that sign.')
    figure('results/pong_motion_input_check.png','Figure 14. Real ALE observations and the two motion channels. Static pixels disappear from both motion channels.',height=147)
    p('A fair comparison starts both value-baseline models fresh: stacked frames versus split frame difference, actor lr=0.0001, critic lr=0.0003, uniform time weighting from episode 1, seed 0, 3,000 episodes and 500 evaluations. Reusing a stacked-frame checkpoint with the new encoding is rejected by the program.')
    rep=load('results/pong_representation_comparison.json')
    table([['Input','500-episode mean +/- SD'],['Stacked frames',score('pong_repr_stack_value')],['Split frame difference',score('pong_repr_motion_value')]],[260,260])
    lo,hi=rep['approximate_paired_ci95']
    p(f"Motion minus stacked score = {rep['motion_minus_stack']:.3f}; approximate paired 95% interval [{lo:.3f}, {hi:.3f}]. The interval includes zero. This experiment does not establish a reliable advantage of motion input at this learning rate.")
    figure('results/pong_representation_comparison.png','Figure 15. Fresh models under matched budgets. A visually higher training segment is not a substitute for the independent score comparison.',height=180)
    page()
    h('12. Pong: actor learning rate is the larger gain')
    p('Because the representation experiment did not resolve weak learning, the next comparison fixed split-frame-difference inputs, the CNN, value baseline, uniform time weighting, critic lr=0.0003, batch=8, clipping=5 and seed 0. The completed fresh lr=0.0001 model was reused unchanged; fresh actors and critics were trained at actor lr=0.0003 and 0.001. Each used 3,000 episodes, exactly 375 actor updates, followed by 500 evaluation episodes.')
    lr=load('results/pong_learning_rate_comparison.json')
    table([['Actor lr','Evaluation mean +/- SD','Gain over 0.0001'],*[ [x['actor_lr'],f"{x['evaluation_mean']:.3f} +/- {x['evaluation_sample_sd']:.3f}",'Reference' if x['actor_lr']==.0001 else f"{next(d['mean_gain_vs_control'] for d in lr['differences'] if d['actor_lr']==x['actor_lr']):.3f}"] for x in lr['runs']]],[100,240,180])
    figure('results/pong_learning_rate_comparison.png','Figure 16. Matched training curves. The actor lr=0.001 configuration learns much faster within the fixed 3,000-episode budget.',height=240)
    d=next(x for x in lr['differences'] if x['actor_lr']==.001);lo,hi=d['approximate_paired_ci95']
    p(f"The lr=0.001 gain is {d['mean_gain_vs_control']:.3f} score units, with approximate paired rollout interval [{lo:.3f}, {hi:.3f}]. This is much larger than the earlier budget, weighting and representation gains. It is evidence for the tested actor learning rate within this specific practical configuration, not proof of its effect under the strict outer-discounted objective or every network.")
    p('The evaluation data here were used to select lr=0.001. An independent confirmation is therefore needed before presenting the selected seed as representative; section 13 provides that check. A new no-baseline counterpart under this improved configuration has not been run, so its improvement is not attributed to the baseline alone.')
    page()
    required_page('12a. Improved Pong extension: selected seed 0','pong_motion_lr001_value')
    p('Supplementary uniform-time-weighted method, not the strict-objective core experiment. The selected policy scores -7.742 +/- 6.311; 60 of 500 evaluation scores are positive, and the maximum is +12. The mean remains negative, so the policy is not claimed to win reliably. Selection on these results makes independent seeds important.')
    page()
    h('13. Independent training-seed confirmation')
    p('The learning rate was selected on seed 0. Two additional models were then initialized and trained independently at seeds 1 and 2, with unchanged hyperparameters and the same 3,000-episode budget. No optimizer or trained representation is transferred. The program seed changes initialization, sampling and disjoint environment-seed ranges; each model receives 500 evaluations and a seed-matched random reference.')
    rows=[['Training seed','Role','Mean +/- rollout SD','Positive scores / 500']]
    for x in seed['runs']:rows.append([x['seed'],'Selection' if x['seed']==0 else 'Confirmation',f"{x['evaluation_mean']:.3f} +/- {x['evaluation_sample_sd']:.3f}",x['positive_score_episodes']])
    table(rows,[85,105,220,110])
    figure('results/pong_seed_validation.png','Figure 19. Individual training curves and independent 500-episode score means. Seed 0 was used for hyperparameter selection.',height=213)
    p(f"The independent seed-1/seed-2 mean is {seed['independent_confirmation_mean']:.3f}. The three-seed mean is {seed['three_seed_mean']:.3f}; the sample SD between the three policy means is {seed['sample_sd_between_three_seed_means']:.3f}. These describe training-procedure variability separately from each policy's rollout SD.")
    p('The conclusion is limited: both confirmation runs improve substantially over their random references, supporting repeatable learning of the selected configuration. Only three training seeds are available, and average scores remain negative. No new low-learning-rate model was trained at seeds 1 and 2, so this is not a complete learning-rate-by-seed interaction study. Pooling 1,500 scores would not create 1,500 independent training runs.')
    p('Training seed ranges: 20,000,000 + 100,000*seed through that value +2,999. Evaluation ranges: 4,000,000 + 10,000*seed through that value +499. Training, evaluation and separate diagnostic ranges are disjoint for these reported runs.',small=True)
    page()
    h('14. Shuttle validation beyond basic unit tests')
    p('The handout asks for predicted algorithm behavior and quantitative evidence. The following tests compare different algorithms, independently enumerate small problems, change meaningful parameters, and compare DP with independent simulation. They are not merely file-format checks.')
    table([['Prediction / test','Observed result'],['Tiny problem: 5 states, 2 arrivals, horizon 4; recursive action/outcome tree vs DP',f"Maximum error {shuttle['tiny_tree_enumeration_max_error']:.1f}"],['VI and PI must agree',f"Value max error {shuttle['vi_vs_pi_max_error']:.3e}; 0 policy mismatches"],['PI must satisfy Bellman optimality',f"Residual {shuttle['policy_evaluation_bellman_residual']:.3e}"],['T=500 approximates discounted infinite horizon',f"Value max error {shuttle['finite_vs_infinite_max_error']:.3e}"],['Free dispatch should serve positive queues','All positive states dispatch'],['Zero capacity should make paid dispatch useless','No state dispatches'],['gamma=0 should remove future effects','All values equal minimum immediate cost']],[295,225])
    sim=load('results/shuttle/simulation_validation.json')
    sub('Independent Monte Carlo cost validation')
    rows=[['Initial queue','DP cost','Simulation mean','Standard error']]
    for x in sim['rows']:rows.append([x['initial_state'],fnum(x['dp_cost']),fnum(x['simulated_mean']),fnum(x['standard_error'])])
    table(rows,[105,140,155,120])
    p('For each queue, 40,000 independent 500-step simulations use seed 314159. Four of five values fall in marginal 95% intervals; state 15 differs by -2.44 standard errors and is retained. All five fall in the simultaneous 95% Bonferroni intervals. The maximum omitted discounted tail is below 6.9e-8. Agreement within quantified simulation error supports the transition/cost implementation; it does not validate arbitrary overflow assumptions.')
    code('python shuttle.py --out runs/shuttle\npython validate_shuttle_simulation.py')
    page()
    h('15. Policy-gradient validation and diagnostics')
    sub('Independent gradients and baseline variance')
    p('A two-step Bernoulli control problem enumerates all four action sequences and compares the analytic discounted-objective derivative with the expected score-function estimator. Both zero and nonzero constant baselines have error below 6e-8. A CartPole-network parameter also matches a central finite difference to about 4.2e-12. These checks establish the implemented estimator on a problem with a known derivative, not merely that training executes.')
    rows=[['Frozen actor / held-out samples','Plain variance trace','Baseline trace','Ratio (95% bootstrap interval)']]
    for game,path in config['validation'].items():
        j=load(path);lo,hi=j['ratio_bootstrap_95_percentile'];rows.append([f"{game}, {j['trajectories']} trajectories",f"{j['plain_gradient_variance_trace']:.5g}",f"{j['baseline_gradient_variance_trace']:.5g}",f"{j['baseline_to_plain_ratio']:.5f} [{lo:.5f}, {hi:.5f}]"])
    table(rows,[155,110,105,150])
    p('Variance is the trace of the sample covariance of full actor gradients. Both estimators use exactly the same held-out trajectories and the same frozen actor; the learned critic is frozen too. Intervals use 300 paired bootstrap resamples. CartPole uses the final 2,000-episode baseline checkpoint and Pong the strict 1,000-episode checkpoint. These measurements do not establish variance reduction for every later method or prove improved game score. The old almost-deterministic Pong run had numerically degenerate variance and is excluded from meaningful ratio interpretation.')
    sub('Problem-specific implementation and environment checks')
    p('The chunked actor/critic loss is checked against an independent dense loss, with and without baseline, under both time weights; the synthetic trajectory crosses the 256-step chunk boundary. Tests also verify critic detachment. Motion tests check sign reconstruction, packed-state round trips, static-pixel cancellation, history reset and agreement between the state used to sample actions and the state later used for the gradient.')
    diag=load('results/pong_control_diagnostic.json')
    response=diag['action_response']
    table([['Real ALE action check','Observed player-paddle response'],['Action 2, three recorded step intervals',f"Center y: {response[0]['paddle_y'][0]:.1f} -> {response[0]['paddle_y'][-1]:.1f} (upward)"],['Action 3, three recorded step intervals',f"Center y: {response[1]['paddle_y'][0]:.1f} -> {response[1]['paddle_y'][-1]:.1f} (downward)"],['Separate 10-episode diagnostic',f"Hand-coded tracking mean {diag['mean_scores']['tracking']:.1f}; random mean {diag['mean_scores']['random']:.1f}"]],[250,270])
    p('These probes confirm movement direction in real frames. The simple tracking controller is weak and is not a trained-policy result. Activation checks on the committed 3,000-episode models found nonzero ReLU activity; the reproducible measurements are in results/pong_endpoint_activations.json. These are network-health checks, not score evidence. Original first-10-step traces remain available for the initial PG runs.')
    code('python policy_gradient.py --test-only\npython reproduce_validation.py\npython -m unittest test_pong_time_weighting test_pong_representation\npython diagnose_cartpole_control.py\npython diagnose_pong_control.py\npython diagnose_pong_activations.py\npython audit_results.py')
    page()
    h('16. Edge cases, objective checks and limits')
    table([['Part / edge case','Handling and evidence'],['Shuttle: s=0, empty dispatch','Never create negative passengers; dispatch still costs 100. Boundary test checks the immediate cost.'],['Shuttle: s=200 and overflow','Waiting yields 200 for every arrival; dispatch yields 186-190. Clipping and overflow assumption are explicit.'],['Shuttle: terminal time and ties','J_500=0; late-horizon actions can differ. Greedy ties choose wait consistently.'],['CartPole: termination or truncation','Both end the sampled episode; no value bootstrap is added. The finite 500-step return is evaluated.'],['Pong: restricted actions and resets','Only actions 2/3; two-frame history resets per episode. Split motion starts with zero change.'],['Sparse rewards / long episodes','Full-episode returns, no score-point reset. Exact gamma^t and uniform variants are explicitly distinguished.'],['Variable trajectory lengths / memory','256-step gradient chunks preserve the sum; Pong averages per episode, not by pooling all steps.'],['Non-finite or extreme gradients','Reject non-finite gradients; batched Pong clips norm at 5. Low-entropy/zero-variance failures are retained.'],['Warm-start continuation','Check episode count against CSV; preserve source prefix, optimizer and CPU sampler RNG; record source hashes.'],['Changing input representation','Reject cross-representation resume; use a fresh network and optimizer for fair comparisons.'],['Baseline fitting leakage','Detach baseline; compute before fitting the current data. Direct variance checks use frozen actor/critic.']],[175,345])
    sub('Training objective and whole-game score can disagree')
    obj=load('results/pong_discounted_objective.json')
    rows=[['Strict-objective policy','Mean discounted return','Mean score']]
    for name,x in obj['summaries'].items():rows.append([name.replace('pong_batch_',''),f"{x['mean_discounted_return']:.4f}",f"{x['mean_score']:.3f}"])
    table(rows,[220,150,150])
    p('This separate diagnostic uses 200 episodes per policy on held-out seeds starting at 6,000,000. Both trained strict-objective policies improve discounted return over random while remaining poor at whole-game score. This demonstrates why the two metrics must be reported separately; it does not identify exactly which reward timings changed. Run python evaluate_pong_objective.py --episodes 200 --device cpu to reproduce.',small=True)
    page()
    h('17. Reproduction, evidence and submission')
    p('Use Python 3.12. The measured game runs used PyTorch 2.14.1, Gymnasium 1.4.0, ALE 0.12.1, NumPy 2.5.4, Adam and one CPU thread. CPU was used for CartPole and historical runs, Apple MPS for batched Pong. The recorded lock file describes the actual training environment. Report rendering is separate from training; hardware and numeric differences can change stochastic trajectories.')
    code('python3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements-lock.txt')
    p('From the repository root, write repeats to fresh runs/ directories. The four commands below correspond exactly to the core report configurations. Every trained policy is evaluated for 500 episodes; --device cpu is available if MPS is unavailable.')
    code('python policy_gradient.py --game cartpole --baseline none --lr 0.001 --episodes 2000 --eval-episodes 500 --out runs/cartpole_none\npython policy_gradient.py --game cartpole --baseline value --lr 0.001 --episodes 2000 --eval-episodes 500 --out runs/cartpole_value\npython pong_batch.py --baseline none --lr 0.0001 --episodes 1000 --batch-episodes 8 --eval-episodes 500 --device mps --out runs/pong_none\npython pong_batch.py --baseline value --lr 0.0001 --episodes 1000 --batch-episodes 8 --eval-episodes 500 --device mps --out runs/pong_value')
    p('The improved seed-0 extension can be repeated without changing source code. For independent confirmation change --seed to 1 or 2 and choose fresh output directories.')
    code('python pong_batch.py --baseline value --lr 0.001 --value-lr 0.0003 --episodes 3000 --eval-episodes 500 --batch-episodes 8 --seed 0 --device mps --time-weighting uniform --representation split_frame_difference --out runs/pong_improved_seed0')
    p('The batch scripts run_learning_rate_comparison.py and run_seed_validation.py preserve completed results and reject duplicate manifests. They are one-time experiment controllers, not report-regeneration commands. For a repeat, use the explicit training command with fresh outputs. Exact historical source snapshots and SHA-256 manifests remain in results/. Atari ROMs and virtual environments are not committed.')
    code('python audit_results.py\npython reproduce_validation.py\npython prepare_report_figures.py\npython build_report.py')
    p('No retraining is needed to regenerate this PDF or repeat the included actor/critic gradient-variance validation capsules. report_config.json selects the core experiments. results/report_evidence.json ties every reported experiment to raw CSV and summary hashes. The report and required plots are in the public repository. Submit assignment1_report.pdf to Canvas; updating GitHub does not itself submit Canvas.')
    page()
    h('18. What three additional months would add')
    table([['Period','Predictions and experiments','Evidence to collect'],['Month 1: isolate causes','Run a no-baseline counterpart to the improved Pong family at matched seeds. Cross strict/uniform weights with actor lr; test constant vs learned baseline. Vary shuttle capacity, cost ratios, queue cap and overflow rule.','Paired seed effects, discounted and undiscounted returns; Bellman residuals; simulated costs and threshold changes.'],['Month 2: quantify robustness','Use at least 10 predeclared training seeds, separate tuning and confirmation seeds, and equal step/update budgets. Test observation noise and initial-state shifts. Study altered Atari sticky actions only as explicitly separate tasks.','Between-seed uncertainty and failure rates; held-out evaluation curves; robustness profiles without conflating environment changes.'],['Month 3: understand learning','Freeze early/mid/late checkpoints to measure baseline gradient variance. Compare architectures and motion encodings. Study nonstationary/heavy-tailed shuttle arrivals. Replicate selected models across CPU/MPS.','Full-gradient covariance traces and baseline calibration; sample and wall-clock efficiency; independent replication and model-assumption sensitivity.']],[115,235,170])
    p('For shuttle sensitivity, the prediction is that increasing dispatch cost tends to delay dispatch, while increasing holding cost tends to accelerate it. Threshold changes and boundary exceptions would be measured rather than assumed. For baselines, predicted variance reduction would be tested directly at each checkpoint, separately from changes in game score.')
    sub('Conclusions and remaining limitations')
    p('The shuttle solution is supported by independent small-tree enumeration, Bellman consistency, controlled parameter changes and large independent simulations. For policy gradients, lower variance is verified directly rather than inferred from smoother rewards. The original Pong failures remain visible; extra training alone and the first motion ablation were not sufficient. A larger actor learning rate in the practical motion-input family produced the largest tested score gain, and independent seeds support repeatable learning. Average Pong performance remains below winning.')
    p('No trained result is extrapolated from a demo or an unrun budget. Exploratory choices and selected seed 0 are distinguished from independent confirmation. Three training seeds are not enough for a broad reliability claim. The improved baseline family lacks a matched no-baseline counterpart, and the uniform surrogate is not the exact start-state objective of the core experiments.')
    sub('Sources and assistance')
    refs=[('R1','Course handout, mlds_assignment1.pdf, pages 1-4. Supplied assignment, due 26 October 2026 at 11:59 pm.',None),('R2','Farama Foundation. CartPole-v1 environment specification.','https://gymnasium.farama.org/environments/classic_control/cart_pole/'),('R3','Farama Foundation. ALE Pong environment and action specification.','https://ale.farama.org/environments/pong/'),('R4','OpenAI Spinning Up. Intro to Policy Optimization: reward-to-go and baselines.','https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html')]
    for key,title,url in refs:p(f"[{key}] {escape(title)}"+(f'<br/><link href="{url}" color="#2563eb">{url}</link>' if url else ''),small=True)
    p('AI assistance was used for implementation, debugging and report preparation. All numerical experiment results are from actual executions and retained artifacts. The author should be able to explain the estimators, assumptions and validation evidence independently.',small=True)
    page()
    h('Appendix A. Complete measurement inventory')
    p('All formal training experiments below have 500 final-policy evaluation episodes. The explicitly named early diagnostic uses only 20 training and 2 evaluation episodes and is not an assignment-sized result. Other short smoke demos are kept separately. Sample SD is over evaluation episodes, not training seeds. Raw curves and histograms are in each result directory; 100-episode averages apply when enough training episodes exist.',small=True)
    rows=[['Result directory (under results/)','Train','Eval','Mean','Sample SD']]
    for x in evidence:rows.append([x['experiment'],x['train_episodes'],x['evaluation_episodes'],fnum(x['mean']),fnum(x['sample_sd'])])
    table(rows,[270,50,45,80,75],compact=True)
    p('Requirement map: shuttle MDP/assumptions and three requested figures - sections 1-2; Part 1 and Part 2 algorithms/baseline - section 3; all four required reward curves and 500-episode histograms - sections 4-7; failure-driven improvements - sections 8-13; quantitative validation - sections 14-15; per-part edge cases - section 16; reproducibility/GitHub/Canvas - section 17; three additional months - section 18.',small=True)
    def footer(canvas,doc):
        canvas.saveState();canvas.setFillColor(MUTED);canvas.setFont('Helvetica',8)
        canvas.drawString(46,25,'Dorrie-ge | MLDS Assignment 1 | Measured experiments')
        canvas.drawRightString(566,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(ROOT/'assignment1_report.pdf'),pagesize=(612,792),leftMargin=46,rightMargin=46,topMargin=39,bottomMargin=43,title='MLDS Assignment 1 - Complete Experimental Report',author='Dorrie-ge')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    readme=ROOT/'README.md'
    start='<!-- COMPLETE_REPORT START -->';end='<!-- COMPLETE_REPORT END -->'
    lines=[start,'## Complete report and final evidence','',
           '[Download the complete PDF report](assignment1_report.pdf). Updated 23 October 2026.',
           'The report covers the initial implementation, retained failures, controlled improvements,',
           'independent seed confirmation, required figures, edge cases, quantitative validation and',
           'the additional three-month experiment plan. All values come from stored measurements.','',
           '### Improved Pong extension: independent training seeds','',
           '| Training seed | Role | 500-episode mean +/- sample SD |','|---|---|---:|']
    lines += [f"| {x['seed']} | {x['role']} | {x['evaluation_mean']:.3f} +/- {x['evaluation_sample_sd']:.3f} |" for x in seed['runs']]
    lines += ['',f"The two independent confirmation runs average {seed['independent_confirmation_mean']:.3f}.",
              f"Across all three policy means: {seed['three_seed_mean']:.3f}, with between-seed sample SD {seed['sample_sd_between_three_seed_means']:.3f}.",
              'Pong learning improved, but mean scores remain negative. Three seeds are a small sample.',
              'The improved extension uses uniform actor time weights; the exact start-state discounted',
              'objective is preserved in the separately reported four core experiments.','',
              '[Requirement-to-evidence map](results/report_requirements.json) and',
              '[raw-data report evidence hashes](results/report_evidence.json) support the report.',
              'Regenerate without training: `python prepare_report_figures.py` then `python build_report.py`.',
              'Submit the PDF to Canvas; a GitHub update alone does not submit the assignment.',end]
    text=readme.read_text()
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+'\n'.join(lines)+after
    else:
        text=text.replace('## Installation','\n'.join(lines)+'\n\n## Installation',1)
    readme.write_text(text)
    print('Built',ROOT/'assignment1_report.pdf','from',len(evidence),'audited experiments')

if __name__=='__main__':build()
