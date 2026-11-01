"""Copy the exact main-branch teaching sources into the static Pages artifact."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


MODULES = (
    'Motion_Models',
    'Sensors_Models',
    'Gaussian_Filters',
    'Discrete_Filters',
    'SLAM',
    'Mapping',
)
SOURCE_DIRS = (*MODULES, '2D_maps')


def export_source_snapshot(source_root: Path, output_root: Path, expected_commit: str | None = None) -> dict:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    actual_commit = subprocess.check_output(
        ['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if expected_commit and actual_commit != expected_commit:
        raise ValueError(f'Main checkout is {actual_commit}, expected {expected_commit}')

    tracked = subprocess.check_output(
        ['git', '-C', str(source_root), 'ls-files', '-z', '--', *SOURCE_DIRS]
    ).split(b'\0')
    paths = sorted(Path(raw.decode()) for raw in tracked if raw)
    found = {path.parts[0] for path in paths}
    missing = set(SOURCE_DIRS) - found
    if missing:
        raise ValueError(f'Main checkout is missing source directories: {sorted(missing)}')

    target_root = output_root / 'source'
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True)
    files = []
    for relative in paths:
        source = source_root / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f'Cannot export tracked source file: {relative}')
        payload = source.read_bytes()
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        files.append({
            'path': relative.as_posix(),
            'module': relative.parts[0],
            'size': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        })

    manifest = {'branch': 'main', 'commit': actual_commit, 'modules': MODULES, 'files': files}
    (target_root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
