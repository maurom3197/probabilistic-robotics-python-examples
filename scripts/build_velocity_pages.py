"""Build a static, backend-free velocity lab from this repository's sampler."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def export_core(source):
    tree = ast.parse(source)
    sampler = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'sample_velocity_motion_model')
    # Keep the actual repository's three noise draws, including their alpha
    # parameterization. Isolate plotting imports and main() from the browser.
    draws = []
    for target in ['v_hat', 'w_hat', 'gamma_hat']:
        matches = [n for n in sampler.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]
        if len(matches) != 1:
            raise ValueError('Sampler structure changed: review the browser adapter')
        draws.append(ast.unparse(matches[0]))
    digest = hashlib.sha256(source.encode()).hexdigest()
    header = f'''# Generated from Motion_Models/velocity_motion_model.py; do not edit.
# Source SHA256: {digest}
# Repository license applies; see LICENSE at the repository root.
import numpy as np
from math import sin, cos


def propagate_velocity(x, v_hat, w_hat, gamma_hat, dt):
    """Same circular-arc model, with the analytic straight-line limit.

    sin(half_turn)/half_turn avoids cancellation/division by zero at omega=0.
    """
    half_turn = 0.5 * w_hat * dt
    sinc = 1.0 if abs(half_turn) < 1e-10 else sin(half_turn) / half_turn
    travel = v_hat * dt * sinc
    return np.array([
        x[0] + travel * cos(x[2] + half_turn),
        x[1] + travel * sin(x[2] + half_turn),
        x[2] + (w_hat + gamma_hat) * dt,
    ])


def sample_velocity_motion_model(x, u, a, dt):
'''
    return header + ''.join('    '+line+'\n' for line in draws) + \
        '    return propagate_velocity(x, v_hat, w_hat, gamma_hat, dt)\n'


def build(source_root=None):
    source_root = Path(source_root or os.environ.get('ROBOTICS_SOURCE_ROOT', ROOT)).resolve()
    source_file = source_root/'Motion_Models/velocity_motion_model.py'
    source = source_file.read_text()
    output = ROOT/'docs'
    output.mkdir(exist_ok=True)
    for path in (ROOT/'web_velocity').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            relative = path.relative_to(ROOT/'web_velocity')
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    (output/'python').mkdir(exist_ok=True)
    (output/'python/velocity_core.py').write_text(export_core(source))
    (output/'provenance.json').write_text(json.dumps(dict(
        source='Motion_Models/velocity_motion_model.py',
        branch='main',
        commit=os.environ.get('ROBOTICS_SOURCE_COMMIT'),
        sha256=hashlib.sha256(source.encode()).hexdigest(),
        adaptation='Plot-free export; stable straight-line limit; original alpha noise draws retained',
    ), indent=2)+'\n')
    (output/'.nojekyll').touch()
    print('Static lab generated in docs/')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', action='store_true', help='Copy Pyodide, NumPy and Three.js into docs/vendor')
    parser.add_argument('--source-root', type=Path, help='Separate checkout of the main branch')
    args = parser.parse_args()
    build(args.source_root)
    if args.vendor:
        from vendor_velocity_runtime import vendor
        vendor(ROOT/'docs')
