"""Copy pinned runtime assets into docs/ for hosting entirely on GitHub Pages.

Uses only Python's standard library. Run build_velocity_pages.py first.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/'
THREE = 'https://cdn.jsdelivr.net/npm/three@0.180.0/'


def dependency_closure(packages, requested):
    found = set()
    def visit(name):
        if name in found:
            return
        if name not in packages:
            raise ValueError(f'Package absent from Pyodide lockfile: {name}')
        found.add(name)
        for dependency in packages[name].get('depends', []):
            visit(dependency)
    for name in requested:
        visit(name)
    return sorted(found)


def download(url, target, expected_sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.part')
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as out:
            while chunk := response.read(1024*1024):
                out.write(chunk); digest.update(chunk)
        if expected_sha and digest.hexdigest() != expected_sha:
            raise ValueError(f'Checksum mismatch: {target.name}')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'{target.name}: {target.stat().st_size/1024**2:.2f} MiB')
    return dict(url=url, sha256=digest.hexdigest(), bytes=target.stat().st_size)


def vendor(output):
    output = Path(output)
    records = {}
    pyroot, threeroot = output/'vendor/pyodide', output/'vendor/three'
    lockfile = pyroot/'pyodide-lock.json'
    records['pyodide-lock.json'] = download(PYODIDE+'pyodide-lock.json', lockfile)
    packages = json.loads(lockfile.read_text())['packages']
    for name in ['pyodide.mjs', 'pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip']:
        records[name] = download(PYODIDE+name, pyroot/name)
    for name in dependency_closure(packages, ['numpy']):
        package = packages[name]
        records[name] = download(PYODIDE+package['file_name'], pyroot/package['file_name'], package.get('sha256'))
    for name in ['three.module.js', 'three.core.js']:
        records[name] = download(THREE+'build/'+name, threeroot/name)
    records['three-license'] = download(THREE+'LICENSE', threeroot/'LICENSE')
    records['pyodide-license'] = download('https://raw.githubusercontent.com/pyodide/pyodide/0.27.7/LICENSE', pyroot/'LICENSE')
    (output/'vendor/manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    # Switch only after every download succeeds. All paths include the project
    # subdirectory automatically through import.meta.url.
    (output/'runtime-config.js').write_text(
        "export const PYODIDE_BASE = new URL('./vendor/pyodide/', import.meta.url).href;\n"
        "export const THREE_MODULE = new URL('./vendor/three/three.module.js', import.meta.url).href;\n")
    total = sum(p.stat().st_size for p in (output/'vendor').rglob('*') if p.is_file())
    print(f'Runtime copied: {total/1024**2:.1f} MiB. All application resources now come from this Pages site.')


if __name__ == '__main__':
    vendor(Path(__file__).resolve().parents[1]/'docs')
