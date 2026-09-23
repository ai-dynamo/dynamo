from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923')
repo=root/'src/dynamo'
old=root.parent/'dynamo-tyche-hf-runtime-m2048-20260917/src/dynamo'
dep=repo/'campaign/deps/zmq'
shutil.copytree(old/'campaign/deps/zmq',dep)
p=repo/'lib/bindings/python/Cargo.toml'
p.write_text(p.read_text()+'\n# Benchmark-only ZMQ socket capacity; no other dependency changes.\n[patch.crates-io]\nzmq = { path = "../../../campaign/deps/zmq" }\n')
p=repo/'lib/bindings/python/Cargo.lock';s=p.read_text();chunks=s.split('[[package]]')
for i,c in enumerate(chunks):
    if c.startswith('\nname = "zmq"\n'):
        chunks[i]='\n'.join(l for l in c.split('\n') if not l.startswith(('source =','checksum =')))
p.write_text('[[package]]'.join(chunks))
p=repo/'lib/runtime/src/transports/event_plane/zmq_transport.rs';s=p.read_text()
prior=(old/'lib/runtime/src/transports/event_plane/zmq_transport.rs').read_text()
start=prior.index('    // Campaign-only capacity setting')
end=prior.index('    // Configure the process-wide context',start)
assert s.count('    let context = Context::new();\n')==1
s=s.replace('    let context = Context::new();\n','    let context = Context::new();\n'+prior[start:end]);p.write_text(s)
(root/'manifests/benchmark-overlay.patch').write_bytes(subprocess.check_output(['git','-C',str(repo),'diff']))
changed=subprocess.check_output(['git','-C',str(repo),'diff','--name-only'],text=True).splitlines()
files={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in changed}
for p in (root/'configs').glob('*.json'):
    if p.name in ('template.json', 'order.json'):continue
    c=json.loads(p.read_text());c['pins']['dirty_source_files']=files;p.write_text(json.dumps(c,indent=2)+'\n')
(root/'manifests/overlay-hashes.json').write_text(json.dumps(files,indent=2)+'\n')
(repo/'.venv/bin').mkdir(parents=True)
py=root.parent/'dynamo-tyche-sidecar-agentx-c1024-dp1-20260910/src/dynamo/.venv/bin/python'
p=repo/'.venv/bin/python'
p.write_text(f'''#!/usr/bin/env bash
set -e
unset RAYON_NUM_THREADS RAYON_RS_NUM_THREADS TOKENIZERS_PARALLELISM FASTOKENS_BPE_THREADS
export PYTHONPATH="{repo}/lib/bindings/python/src:{repo}/components/src"
exec unshare --user --map-root-user --mount "{root}/harness/frontend_cache.sh" "{py}" "$@"
''');p.chmod(0o755)
(root/'control/SOURCE_READY').write_text(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True))
