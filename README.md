# MLDS Assignment 1: Reinforcement Learning

Dorrie Ge

## Where to find things

| Location | Contents |
|---|---|
| [reports/](reports/) | Submission PDF and experiment-selection configuration |
| [src/](src/) | Training, validation, experiment controllers, tests and report generation |
| [results/](results/) | Measured CSVs, figures, summaries, model weights and evidence manifests |
| [logs/](logs/) | Recorded training logs |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Detailed methods, historical failures and improvement comparisons |

<!-- COMPLETE_REPORT START -->
## Complete report and final evidence

[Download the complete PDF report](reports/assignment1_report.pdf). 
The report covers the initial implementation, retained failures, controlled improvements,
independent seed confirmation, required figures, edge cases, quantitative validation and
the additional three-month experiment plan. 

### Improved Pong extension: independent training seeds

| Training seed | Role | 500-episode mean +/- sample SD |
|---|---|---:|
| 0 | selection_run | -7.742 +/- 6.311 |
| 1 | independent_confirmation | -9.250 +/- 3.668 |
| 2 | independent_confirmation | -10.032 +/- 3.796 |

The two independent confirmation runs average -9.641.
Across all three policy means: -9.008, with between-seed sample SD 1.164.
Pong learning improved, but mean scores remain negative. 
The improved extension uses uniform actor time weights; the exact start-state discounted
objective is preserved in the separately reported four core experiments.

[Requirement-to-evidence map](results/report_requirements.json) and
[raw-data report evidence hashes](results/report_evidence.json) support the report.
Regenerate without training: `python src/prepare_report_figures.py` then `python src/build_report.py`.
Submit the PDF to Canvas; a GitHub update alone does not submit the assignment.
<!-- COMPLETE_REPORT END -->

## Installation

Run commands from the repository root. Python 3.12 was used.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

The lock file records the measured macOS environment. On another platform use `requirements.txt` if needed and record version changes. Atari ROMs and virtual environments are excluded from Git.

## Reproduce the required experiments

The following commands train fresh models and evaluate each trained policy for **500 episodes**. Use a fresh output directory for each repeat. MPS was used for batched Pong; `--device cpu` is also supported.

```bash
python src/shuttle.py --out runs/shuttle
python src/policy_gradient.py --game cartpole --baseline none --lr 0.001 --episodes 2000 --eval-episodes 500 --out runs/cartpole_none
python src/policy_gradient.py --game cartpole --baseline value --lr 0.001 --episodes 2000 --eval-episodes 500 --out runs/cartpole_value
python src/pong_batch.py --baseline none --lr 0.0001 --episodes 1000 --batch-episodes 8 --eval-episodes 500 --device mps --out runs/pong_none
python src/pong_batch.py --baseline value --lr 0.0001 --episodes 1000 --batch-episodes 8 --eval-episodes 500 --device mps --out runs/pong_value
```

For the improved Pong extension, use the following command. Independent confirmation uses seeds 1 and 2 with separate output directories. Uniform actor time weights change the strict objective estimator; the report explains the distinction.

```bash
python src/pong_batch.py --baseline value --lr 0.001 --value-lr 0.0003 --episodes 3000 --eval-episodes 500 --batch-episodes 8 --seed 0 --device mps --time-weighting uniform --representation split_frame_difference --out runs/pong_improved_seed0
```

## Validate and regenerate without training

```bash
python src/audit_results.py
python src/policy_gradient.py --test-only
python -m unittest discover -s src -p 'test_pong_*.py'
python src/reproduce_validation.py
python src/prepare_report_figures.py
python src/build_report.py
```

`audit_results.py` recomputes statistics from every experiment's raw CSVs. `reproduce_validation.py` checks committed actor/critic capsule hashes and repeats analytic-gradient and frozen-policy gradient-variance experiments. The tests also check predicted time-weighting behavior, motion-input encoding and dense-versus-chunked gradients. See [the report](reports/assignment1_report.pdf), sections 14-16, and [the experiment record](EXPERIMENTS.md) for the problem-specific validation evidence and optional simulator/control diagnostics.

Report regeneration reads committed measurements selected by [reports/report_config.json](reports/report_config.json); it does not launch training. The one-time `run_*_comparison.py` / `run_seed_validation.py` controllers retain manifests and reject duplicate experiment runs. Use the explicit commands above for fresh repeats.

