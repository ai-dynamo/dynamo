import base64,csv,hashlib,importlib.metadata as m,io,json,sys
from pathlib import Path
r=Path('/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923')
result={'python':sys.version,'executable':sys.executable,'packages':{}}
for name in ['aiperf','transformers','tokenizers','pyzmq','uvloop','orjson']:
    try:d=m.distribution(name)
    except m.PackageNotFoundError:continue
    raw=d.read_text('direct_url.json')
    x={'version':d.version,'location':str(d.locate_file('')),'direct_url':json.loads(raw) if raw else None}
    if name=='aiperf':
        changed=[];count=0
        for file,h,size in csv.reader(io.StringIO(d.read_text('RECORD'))):
            if not h:continue
            algorithm,digest=h.split('=',1)
            path=d.locate_file(file);count+=1
            found=base64.urlsafe_b64encode(hashlib.new(algorithm,path.read_bytes()).digest()).decode().rstrip('=')
            if found!=digest:changed.append(file)
        x.update(verified_files=count,changed_files=changed)
        assert not changed,changed
    result['packages'][name]=x
(r/'manifests/client-provenance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
