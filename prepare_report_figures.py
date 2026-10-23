"""Prepare report-only graphics from measured CSV/JSON, without training."""
from pathlib import Path
import os,json
import numpy as np
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    fig,ax=plt.subplots(figsize=(8,3.4))
    for baseline,label in [('none','No baseline'),('value','Value baseline')]:
        means=[]
        for prefix in ['pong_batch','pong_budget3000','pong_budget5000']:
            s=json.loads((ROOT/f'results/{prefix}_{baseline}/summary.json').read_text());means.append(s['mean'])
        ax.plot([1000,3000,5000],means,'o-',label=label)
    ax.axhline(-20.514,color='#94a3b8',linestyle='--',label='Random reference')
    ax.set(xlabel='Total training episodes',ylabel='500-episode evaluation mean',title='Strict-objective Pong: changing only training budget')
    ax.set_xticks([1000,3000,5000]);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/report_budget_comparison.png',dpi=180);plt.close(fig)
    data=np.genfromtxt(ROOT/'results/shuttle/values.csv',delimiter=',',names=True)
    for field,name,title,ylabel in [('finite_cost_t0','finite_value','Finite-horizon cost at t=0, T=500','Expected discounted cost'),('infinite_cost_vi','infinite_value','Infinite-horizon cost, gamma=0.95','Expected discounted cost'),('dispatch_policy','optimal_policy','Optimal dispatch policy','Action: 0=wait, 1=dispatch')]:
        fig,ax=plt.subplots(figsize=(8.5,2.5))
        if field=='dispatch_policy':ax.step(data['state'],data[field],where='post');ax.set_yticks([0,1]);ax.set_ylim(-.1,1.15)
        else:ax.plot(data['state'],data[field])
        ax.set(xlabel='Customers waiting before dispatch',ylabel=ylabel,title=title);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/f'results/shuttle/{name}.png',dpi=180);plt.close(fig)
    print('Prepared report graphics from measured results.')
if __name__=='__main__':main()
