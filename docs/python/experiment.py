"""A bounded, seeded Python simulation called explicitly by the browser worker."""
import json
import math
import numpy as np
from velocity_core import sample_velocity_motion_model, propagate_velocity

_job = None


def validate(p):
    required = ['x', 'y', 'theta', 'v', 'omega', 'dt', 'steps', 'samples', 'seed', 'noise_mode']
    if any(key not in p for key in required):
        raise ValueError('Incomplete parameters')
    continuous = ['x', 'y', 'theta', 'v', 'omega', 'dt']
    noise_keys = [f'a{i}' for i in range(1, 7)] if p['noise_mode'] == 'alpha' else ['sigma_v', 'sigma_omega', 'sigma_gamma']
    if p['noise_mode'] not in ['alpha', 'sigma']:
        raise ValueError('Invalid noise mode')
    for key in continuous+noise_keys:
        value = p.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid value: {key}')
    if any(p[k] < 0 or p[k] > 10 for k in noise_keys):
        raise ValueError('Noise must be between 0 and 10')
    for key, low, high in [('steps', 1, 60), ('samples', 50, 5000), ('seed', 0, 2147483647)]:
        if isinstance(p[key], bool) or not isinstance(p[key], (int, float)) or int(p[key]) != p[key] or not low <= p[key] <= high:
            raise ValueError(f'{key}: enter an integer between {low} and {high}')
    if not .001 <= p['dt'] <= 3 or abs(p['v']) > 10 or abs(p['omega']) > 10 or abs(p['x']) > 100 or abs(p['y']) > 100 or abs(p['theta']) > math.pi+.001:
        raise ValueError('Parameters are outside the demo limits')
    return p


def _frame():
    j = _job; p = j['particles']; mean = p.mean(axis=0)
    mean[2] = np.arctan2(np.sin(p[:, 2]).mean(), np.cos(p[:, 2]).mean())
    cov = np.cov(p[:, :2], rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    angles = np.linspace(0, 2*np.pi, 65)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    ellipse = (vectors @ (np.sqrt(np.maximum(values, 0)*5.991)[:, None]*circle)).T + mean[:2]
    return dict(step=j['step'], time=j['step']*j['p']['dt'],
                cloud=p[j['shown']].ravel().tolist(), mean=mean.tolist(),
                nominal=j['nominal'].tolist(), ellipse=ellipse.ravel().tolist())


def start_job(parameters_json):
    global _job
    p = validate(json.loads(parameters_json))
    np.random.seed(int(p['seed']))
    pose = np.array([p['x'], p['y'], p['theta']], dtype=float)
    n = int(p['samples'])
    _job = dict(p=p, particles=np.repeat(pose[None, :], n, axis=0),
                nominal=pose.copy(), step=0,
                shown=np.linspace(0, n-1, min(n, 1500), dtype=int))
    return json.dumps(_frame(), allow_nan=False)


def advance_job():
    j = _job; p = j['p']
    if j['step'] >= p['steps']:
        raise ValueError('Simulation already completed')
    u = np.array([p['v'], p['omega']]); dt = p['dt']
    if p['noise_mode'] == 'alpha':
        a = np.array([p[f'a{i}'] for i in range(1, 7)])
        j['particles'] = np.array([sample_velocity_motion_model(x, u, a, dt) for x in j['particles']])
    else:
        sigmas = np.array([p['sigma_v'], p['sigma_omega'], p['sigma_gamma']])
        commands = np.array([p['v'], p['omega'], 0.]) + np.random.normal(size=(len(j['particles']), 3))*sigmas
        j['particles'] = np.array([propagate_velocity(x, *command, dt) for x, command in zip(j['particles'], commands)])
    j['nominal'] = propagate_velocity(j['nominal'], p['v'], p['omega'], 0., dt)
    j['step'] += 1
    return json.dumps(_frame(), allow_nan=False)


def finish_job():
    j = _job; particles = j['particles']; p = j['p']
    heading_errors = (particles[:, 2]-j['nominal'][2]+np.pi)%(2*np.pi)-np.pi
    hist, edges = np.histogram(heading_errors, bins=32)
    resultant = np.hypot(np.sin(particles[:, 2]).mean(), np.cos(particles[:, 2]).mean())
    sigma_theta = np.sqrt(max(0., -2*np.log(np.clip(resultant, 1e-15, 1.))))
    std = np.std(particles[:, :2], axis=0)
    mean = _frame()['mean']
    return json.dumps(dict(parameters=p, sample_count=int(p['samples']), shown_count=len(j['shown']),
                           mean=mean, sigma_x=float(std[0]), sigma_y=float(std[1]), sigma_theta=float(sigma_theta),
                           histogram=dict(counts=hist.tolist(), edges=edges.tolist()),
                           final_particles=particles.tolist()), allow_nan=False)
