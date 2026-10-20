"""Verify reported counts and summary statistics against stored raw measurements."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parent
    audited=[]
    for file in sorted((root/'results').glob('*/summary.json')):
        directory=file.parent
        j=json.loads(file.read_text())
        train=np.atleast_1d(np.genfromtxt(directory/'training.csv',delimiter=',',names=True))
        evaluation=np.atleast_1d(np.loadtxt(directory/'evaluation.csv',skiprows=1))
        random=np.atleast_1d(np.loadtxt(directory/'random_evaluation.csv',skiprows=1))
        assert len(train)==j['episodes'],directory
        assert len(evaluation)==len(random)==j['evaluation_episodes'],directory
        assert np.isfinite(evaluation).all(),directory
        assert np.isclose(evaluation.mean(),j['mean']),directory
        assert np.isclose(evaluation.std(ddof=1),j['std_sample']),directory
        assert np.isclose(random.mean(),j['random_mean']),directory
        if j['game']=='cartpole':
            assert np.all((evaluation>=1)&(evaluation<=500)),directory
        else:
            assert np.all((evaluation>=-21)&(evaluation<=21)),directory
        assert np.allclose(evaluation,np.round(evaluation)),directory
        audited.append(dict(experiment=directory.name,training_episodes=len(train),
                            evaluation_episodes=len(evaluation),mean=float(evaluation.mean()),
                            sample_sd=float(evaluation.std(ddof=1))))
    main_names=json.loads((root/'report_config.json').read_text())['main_experiments']
    for name in main_names:
        assert next(x for x in audited if x['experiment']==name)['evaluation_episodes']==500
    (root/'results/audit.json').write_text(json.dumps(dict(passed=True,experiments=audited),indent=2))
    print('Verified',len(audited),'experiments against raw CSV counts, means, and standard deviations.')


if __name__=='__main__':main()
