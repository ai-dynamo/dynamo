FROM dynamoci.azurecr.io/ai-dynamo/dynamo:multinode-failover-ae51ca3f1-vllm-runtime

COPY <<'PYEOF' /tmp/patch.py
path = '/opt/dynamo/venv/lib/python3.12/site-packages/gpu_memory_service/integrations/vllm/patches.py'
with open(path) as f:
    content = f.read()

old = '        free_bytes = torch.cuda.mem_get_info()[0]'
new = """        free_bytes = torch.cuda.mem_get_info()[0]
        import subprocess as _sp
        _nv = _sp.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader'], capture_output=True, text=True)
        logger.info('[Shadow] BARRIER: torch free=%.2f GiB, nvidia-smi=[%s], needed=%.2f GiB', free_bytes/(1<<30), _nv.stdout.strip(), needed_bytes/(1<<30))"""

if old in content and 'BARRIER: torch' not in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print('Patched')
else:
    print('Already patched or mismatch')
PYEOF

RUN python3 /tmp/patch.py && rm /tmp/patch.py
