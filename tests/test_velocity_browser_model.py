"""Numerical checks using the real repository sampler and the exported model."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_velocity_pages import build, export_core
from vendor_velocity_runtime import dependency_closure

SOURCE_ROOT = Path(os.environ.get('ROBOTICS_SOURCE_ROOT', ROOT)).resolve()
SOURCE = (SOURCE_ROOT/'Motion_Models/velocity_motion_model.py').read_text()
EXPORTED = {}
exec(export_core(SOURCE), EXPORTED)
ORIGINAL = {'np': np}
from math import sin, cos
ORIGINAL.update(sin=sin, cos=cos)
tree = ast.parse(SOURCE)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_velocity_motion_model')
exec(ast.unparse(function), ORIGINAL)


def experiment():
    build()
    sys.path.insert(0, str(ROOT/'docs/python'))
    spec = importlib.util.spec_from_file_location('browser_experiment_test', ROOT/'docs/python/experiment.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def parameters(**kwargs):
    p = dict(x=2., y=4., theta=0., v=.8, omega=.6, dt=.5, steps=12, samples=100, seed=42,
             noise_mode='alpha', a1=.001, a2=.01, a3=.1, a4=.2, a5=.05, a6=.05,
             sigma_v=.1, sigma_omega=.1, sigma_gamma=.02)
    p.update(kwargs); return p


def run(module, p):
    frames = [json.loads(module.start_job(json.dumps(p)))]
    for _ in range(int(p['steps'])):
        frames.append(json.loads(module.advance_job()))
    result = json.loads(module.finish_job())
    result['frames'] = frames
    return result


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.module = experiment()

    def test_export_preserves_repository_samples(self):
        for omega in [-1., .6, 1.5]:
            x = np.array([2., 4., .3]); u = np.array([.8, omega]); a = np.array([.001, .01, .1, .2, .05, .05])
            np.random.seed(42)
            expected = np.array([ORIGINAL['sample_velocity_motion_model'](x, u, a, .5) for _ in range(100)])
            np.random.seed(42)
            actual = np.array([EXPORTED['sample_velocity_motion_model'](x, u, a, .5) for _ in range(100)])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_straight_stationary_reverse_and_tiny_turn(self):
        for v in [0., .8, -.8]:
            for omega in [0., 1e-14, -1e-14]:
                result = run(self.module, parameters(v=v, omega=omega, noise_mode='sigma', sigma_v=0., sigma_omega=0., sigma_gamma=0., steps=2))
                np.testing.assert_allclose(result['final_particles'], np.tile([2.+v, 4., omega], (100, 1)), atol=1e-10)
                self.assertTrue(np.isfinite(result['sigma_theta']))

    def test_deterministic_and_seed_sensitive(self):
        first = run(self.module, parameters())
        self.assertEqual(first, run(self.module, parameters()))
        self.assertNotEqual(first['final_particles'], run(self.module, parameters(seed=43))['final_particles'])

    def test_sigma_increases_spread(self):
        zero = run(self.module, parameters(noise_mode='sigma', sigma_v=0., sigma_omega=0., sigma_gamma=0.))
        noisy = run(self.module, parameters(noise_mode='sigma'))
        self.assertLess(zero['sigma_x'], 1e-12)
        self.assertGreater(noisy['sigma_x'], .01)

    def test_frame_timing_and_all_particle_statistics(self):
        r = run(self.module, parameters(samples=1600, steps=2))
        self.assertEqual(len(r['frames']), 3)
        self.assertEqual([f['time'] for f in r['frames']], [0., .5, 1.])
        self.assertEqual(r['shown_count'], 1500)
        self.assertEqual(len(r['final_particles']), 1600)
        p = np.array(r['final_particles'])
        np.testing.assert_allclose(r['mean'][:2], p[:, :2].mean(axis=0))
        self.assertEqual(sum(r['histogram']['counts']), 1600)
        json.dumps(r, allow_nan=False)

    def test_invalid_inputs(self):
        for patch in [dict(samples=0), dict(steps=1.5), dict(seed=-1), dict(dt=0), dict(a1=-.01), dict(v=float('inf')), dict(noise_mode='invalid')]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.module.start_job(json.dumps(parameters(**patch)))

    def test_vendor_dependency_graph(self):
        packages = {'numpy': {'depends': ['openblas']}, 'openblas': {'depends': []}, 'unused': {'depends': []}}
        self.assertEqual(dependency_closure(packages, ['numpy']), ['numpy', 'openblas'])
        with self.assertRaises(ValueError): dependency_closure(packages, ['missing'])


if __name__ == '__main__': unittest.main()
