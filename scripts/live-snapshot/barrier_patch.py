import sys
path = '/opt/dynamo/venv/lib/python3.12/site-packages/gpu_memory_service/integrations/vllm/patches.py'
with open(path) as f:
    c = f.read()
if 'BARRIER_PATCHED' not in c:
    # Replace the entire barrier section in allocate_kv_cache_on_wake
    old = """        free_bytes = torch.cuda.mem_get_info()[0]
        if free_bytes < needed_bytes:
            logger.info(
                "[Shadow] Waiting for GPU memory before KV cache allocation "
                "(need %.2f GiB, free %.2f GiB)",
                needed_bytes / (1 << 30),
                free_bytes / (1 << 30),
            )
            while free_bytes < needed_bytes:
                time.sleep(0.5)
                free_bytes = torch.cuda.mem_get_info()[0]
            logger.info(
                "[Shadow] GPU memory available (free %.2f GiB), proceeding",
                free_bytes / (1 << 30),
            )"""
    new = """        free_bytes, total_bytes = torch.cuda.mem_get_info()
        needed_bytes = int(0.7 * total_bytes)  # PATCHED: require 70% free
        import subprocess as _sp, time as _time
        _nv = _sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"], capture_output=True, text=True)
        _procs = _sp.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"], capture_output=True, text=True)
        logger.info("[BARRIER_PATCHED] needed=%.2f GiB (70%% of %.2f), torch_free=%.2f GiB, nvidia_smi=[%s], gpu_procs=[%s], will_block=%s",
            needed_bytes/(1<<30), total_bytes/(1<<30), free_bytes/(1<<30), _nv.stdout.strip(), _procs.stdout.strip().replace(chr(10),"; "), free_bytes < needed_bytes)
        _barrier_start = _time.monotonic()
        if free_bytes < needed_bytes:
            logger.info(
                "[BARRIER_PATCHED] Waiting for GPU memory "
                "(need %.2f GiB, free %.2f GiB)",
                needed_bytes / (1 << 30),
                free_bytes / (1 << 30),
            )
            while free_bytes < needed_bytes:
                time.sleep(0.5)
                free_bytes = torch.cuda.mem_get_info()[0]
                logger.info("[BARRIER_PATCHED] waiting... free=%.2f GiB, elapsed=%.1fs", free_bytes/(1<<30), _time.monotonic()-_barrier_start)
            logger.info(
                "[BARRIER_PATCHED] GPU memory available (free %.2f GiB), waited %.1fs",
                free_bytes / (1 << 30), _time.monotonic()-_barrier_start,
            )"""
    if old in c:
        c = c.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(c)
        print('BARRIER PATCH APPLIED')
    else:
        print('BARRIER PATCH: pattern not found, dumping context...')
        # Debug: show what the barrier area looks like
        import re
        m = re.search(r'free_bytes = torch\.cuda\.mem_get_info.*?proceeding', c, re.DOTALL)
        if m:
            print(m.group()[:500])
        else:
            print('Could not find barrier code at all')
else:
    print('BARRIER PATCH: already applied')
