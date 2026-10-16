"""Shuttle MDP: Python 3 + NumPy. Run: python shuttle.py --out shuttle_results"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np


def model(cap=200, capacity=15, dispatch=100., holding=2., arrivals=(1, 2, 3, 4, 5)):
    states = np.arange(cap + 1)
    left = np.maximum(states[:, None] - capacity * np.arange(2), 0)
    cost = holding * left + dispatch * np.arange(2)
    nxt = np.minimum(left[:, :, None] + np.array(arrivals), cap).astype(int)
    return cost, nxt


def qvalues(v, cost, nxt, gamma):
    return cost + gamma * v[nxt].mean(axis=2)


def finite(cost, nxt, gamma, horizon):
    v = np.zeros(len(cost))
    for _ in range(horizon):
        q = qvalues(v, cost, nxt, gamma)
        v = q.min(axis=1)
    return v, q.argmin(axis=1)


def value_iteration(cost, nxt, gamma, tolerance=1e-10):
    v = np.zeros(len(cost))
    for count in range(1, 100000):
        new = qvalues(v, cost, nxt, gamma).min(axis=1)
        v = new
        residual = np.max(np.abs(qvalues(v, cost, nxt, gamma).min(axis=1) - v))
        if residual < tolerance:
            return v, qvalues(v, cost, nxt, gamma).argmin(axis=1), count, residual
    raise RuntimeError('Value iteration did not converge')


def policy_iteration(cost, nxt, gamma):
    n = len(cost)
    policy = np.zeros(n, dtype=int)
    for count in range(1, 10000):
        p = np.zeros((n, n))
        for s in range(n):
            np.add.at(p[s], nxt[s, policy[s]], 1 / nxt.shape[2])
        v = np.linalg.solve(np.eye(n) - gamma * p, cost[np.arange(n), policy])
        new = qvalues(v, cost, nxt, gamma).argmin(axis=1)
        if np.array_equal(policy, new):
            return v, policy, count
        policy = new
    raise RuntimeError('Policy iteration did not converge')


def svg(path, values, title, ylabel):
    # Standalone vector plot; no plotting package required.
    w, h, l, r, top, bot = 900, 500, 100, 30, 65, 80
    vmax = max(float(np.max(values)), 1.)
    vmax *= 1.08
    def xy(i, val):
        return l + i / (len(values)-1) * (w-l-r), h-bot-val/vmax*(h-top-bot)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<g font-family="Arial, sans-serif" font-size="15" fill="#1e293b">',
                f'<text x="450" y="30" text-anchor="middle" font-size="21">{title}</text>']
    for tick in range(6):
        val = tick * vmax / 5
        _, y = xy(0, val)
        elements += [f'<path d="M{l},{y} H{w-r}" stroke="#e2e8f0"/>',
                     f'<text x="{l-12}" y="{y+5}" text-anchor="end">{val:.1f}</text>']
    for i in range(0, len(values), max(1,(len(values)-1)//5)):
        x, y = xy(i, 0)
        elements.append(f'<text x="{x}" y="{y+27}" text-anchor="middle">{i}</text>')
    points = ' '.join(f'{x:.2f},{y:.2f}' for x,y in (xy(i,v) for i,v in enumerate(values)))
    elements += [f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>',
                 f'<path d="M{l},{top} V{h-bot} H{w-r}" fill="none" stroke="#334155"/>',
                 '<text x="450" y="480" text-anchor="middle">Customers waiting before dispatch, s</text>',
                 f'<text transform="translate(24,250) rotate(-90)" text-anchor="middle">{ylabel}</text>',
                 '</g></svg>']
    path.write_text('\n'.join(elements))


def validate(cost, nxt, gamma, vf, vi, pi, vp):
    checks = {}
    checks['finite_vs_infinite_max_error'] = float(np.max(np.abs(vf-vi)))
    checks['vi_vs_pi_max_error'] = float(np.max(np.abs(vi-vp)))
    checks['vi_pi_policy_mismatches'] = int(np.sum(pi != qvalues(vi,cost,nxt,gamma).argmin(axis=1)))
    checks['policy_evaluation_bellman_residual'] = float(np.max(np.abs(qvalues(vp,cost,nxt,gamma).min(axis=1)-vp)))
    # Full contingent action-tree enumeration for a tiny MDP, independently of DP.
    tiny_c, tiny_n = model(cap=4, capacity=2, dispatch=3, holding=2, arrivals=(1,2))
    def tree(s, horizon):
        if horizon == 0:
            return 0.
        return min(tiny_c[s,a] + gamma * sum(tree(int(ns),horizon-1) for ns in tiny_n[s,a])/2 for a in (0,1))
    small, _ = finite(tiny_c,tiny_n,gamma,4)
    checks['tiny_tree_enumeration_max_error'] = max(abs(small[s]-tree(s,4)) for s in range(5))
    zc, zn = model(dispatch=0)
    zv,zp,_,_ = value_iteration(zc,zn,gamma)
    checks['free_dispatch_all_positive_states_dispatch'] = bool(np.all(zp[1:]==1))
    gc,gp = finite(cost,nxt,0.,1)
    checks['gamma_zero_equals_immediate_min'] = bool(np.allclose(gc,cost.min(axis=1)))
    kc,kn = model(capacity=0)
    kv,kp,_,_ = value_iteration(kc,kn,gamma)
    checks['zero_capacity_never_dispatch'] = bool(np.all(kp==0))
    checks['boundary_200_wait_next_states'] = nxt[200,0].tolist()
    checks['boundary_200_dispatch_next_states'] = nxt[200,1].tolist()
    checks['empty_station_dispatch_cost'] = float(cost[0,1])
    assert checks['tiny_tree_enumeration_max_error'] < 1e-10
    assert checks['vi_vs_pi_max_error'] < 1e-7
    assert checks['vi_pi_policy_mismatches'] == 0
    assert checks['free_dispatch_all_positive_states_dispatch']
    assert checks['gamma_zero_equals_immediate_min']
    assert checks['zero_capacity_never_dispatch']
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('shuttle_results'))
    out = parser.parse_args().out
    out.mkdir(parents=True,exist_ok=True)
    cost,nxt = model()
    gamma = .95
    vf,pf = finite(cost,nxt,gamma,500)
    vi,pi,iterations,residual = value_iteration(cost,nxt,gamma)
    vp,pp,policy_steps = policy_iteration(cost,nxt,gamma)
    stats = validate(cost,nxt,gamma,vf,vi,pp,vp)
    stats.update(vi_iterations=iterations,vi_residual=residual,pi_iterations=policy_steps,
                 dispatch_states=np.flatnonzero(pp).tolist(),cost_at_zero=float(vi[0]))
    (out/'validation.json').write_text(json.dumps(stats,indent=2))
    with (out/'values.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['state','finite_cost_t0','infinite_cost_vi','infinite_cost_pi','dispatch_policy'])
        writer.writerows(zip(range(201),vf,vi,vp,pp))
    svg(out/'finite_value.svg',vf,'Finite-horizon optimal cost at t = 0 (T = 500)','Expected discounted cost')
    svg(out/'infinite_value.svg',vi,'Infinite-horizon optimal cost (gamma = 0.95)','Expected discounted cost')
    svg(out/'optimal_policy.svg',pp,'Optimal policy: 0 = wait, 1 = dispatch','Dispatch action')
    print(json.dumps({k:v for k,v in stats.items() if k!='dispatch_states'},indent=2))
    print('Dispatch states:',np.flatnonzero(pp))


if __name__=='__main__':
    main()
