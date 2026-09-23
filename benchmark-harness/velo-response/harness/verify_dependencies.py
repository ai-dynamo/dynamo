"""Compare selected built registry sources with the archives pinned by Cargo.lock."""
import hashlib,json,tarfile,tomllib
from pathlib import Path
ROOT=Path('/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923')
meta=json.loads((ROOT/'manifests/grace-bindings-metadata.json').read_text())
lock=tomllib.loads((ROOT/'src/dynamo/lib/bindings/python/Cargo.lock').read_text())
checks={(p['name'],p['version']):p.get('checksum') for p in lock['package']}
selected={'fastokens','tokenizers','dynamo-tokenizers','quinn','quinn-proto','quinn-udp','rmp','rmp-serde','prost','tokio','hyper','axum','serde_json'}
results=[]
for p in meta['packages']:
    if p['name'] not in selected:continue
    d=Path(p['manifest_path']).parent
    archive=d.parents[2]/'cache'/d.parent.name/(d.name+'.crate')
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert checksum==checks[p['name'],p['version']],p['name']
    changed=[];count=0
    with tarfile.open(archive) as tar:
        for f in tar.getmembers():
            if not f.isfile():continue
            rel=Path(f.name).relative_to(d.name);count+=1
            if (d/rel).read_bytes()!=tar.extractfile(f).read():changed.append(str(rel))
    results.append({'name':p['name'],'version':p['version'],'archive_sha256':checksum,'files_verified':count,'changed_files':changed})
    assert not changed,(p['name'],changed)
(ROOT/'manifests/selected-dependency-audit.json').write_text(json.dumps(results,indent=2)+'\n')
print('Verified',len(results),'packages and',sum(x['files_verified'] for x in results),'files')
