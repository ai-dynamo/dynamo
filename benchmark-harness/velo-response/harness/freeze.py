"""Record, then verify, the one source and binary set used by every mode."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--main')
p.add_argument('--velo')
p.add_argument('--verify', action='store_true')
a = p.parse_args()
repo = a.root / 'src/dynamo'
manifest = a.root / 'manifests/frozen-build.json'

def git(*args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

locks = ['Cargo.lock', 'lib/bindings/python/Cargo.lock', 'lib/bindings/kvbm/Cargo.lock']
binary = 'lib/bindings/python/src/dynamo/_core.so'
changed = git('diff', '--name-only').splitlines()
allowed = {'lib/runtime/src/transports/event_plane/zmq_transport.rs',
           'lib/bindings/python/Cargo.toml', 'lib/bindings/python/Cargo.lock'}
assert set(changed) <= allowed, changed
files = sorted(set(locks + changed + [binary, '.dynamo-native-rustflags']))
hashes = {name: digest(repo / name) for name in files}
overlay = {name: hashes[name] for name in changed}
if a.verify:
    frozen = json.loads(manifest.read_text())
    assert frozen['dynamo'] == git('rev-parse', 'HEAD')
    assert frozen['files'] == hashes, 'source, lockfile or binary changed after freeze'
    assert frozen['overlay'] == overlay
else:
    assert a.main and a.velo
    subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', a.main, 'HEAD'], check=True)
    for name in locks:
        assert a.velo in (repo / name).read_text(), name
    frozen = {
        'frozen_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'main': a.main, 'dynamo': git('rev-parse', 'HEAD'), 'velo': a.velo,
        'files': hashes, 'overlay': overlay,
        'features': ['tracing/release_max_level_warn', 'velo-ucx'],
        'rustflags': (repo / '.dynamo-native-rustflags').read_text().strip(),
    }
    manifest.write_text(json.dumps(frozen, indent=2) + '\n')
    (a.root / 'manifests/benchmark-overlay.patch').write_text(git('diff') + '\n')
    for name in locks:
        dest = a.root / 'manifests/frozen-locks' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, dest)
    for path in (a.root / 'configs').glob('*.json'):
        if path.name in ('template.json', 'order.json'):
            continue
        config = json.loads(path.read_text())
        config['pins'].update(dynamo=frozen['dynamo'], velo=a.velo, dirty_source_files=overlay)
        path.write_text(json.dumps(config, indent=2) + '\n')
print(json.dumps({'dynamo': frozen['dynamo'], 'binary_sha256': hashes[binary]}))
