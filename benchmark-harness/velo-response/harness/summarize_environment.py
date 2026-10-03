"""Retain the configuration and hardware evidence used by the four-way report."""
import hashlib
import json
import re
from analyze_campaign import ROOT, run_location


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


order = json.loads((ROOT / 'configs/order.json').read_text())
config = json.loads((ROOT / 'configs/r1-tcp.json').read_text())
settings, runs = {}, {}
for label in (x for x in order if x.startswith('r')):
    job, run = run_location(label)
    static = json.loads((ROOT / 'artifacts' / f'network-{job}-{label}' / 'rank-0/static.json').read_text())
    interface = config['network']['interface']
    irq = {str(i): static['irq_affinity'][str(i)] for i in static['interface_irqs'][interface]}
    nic = {k: v['stdout'] for k, v in static['ethtool'][interface].items()}
    selected = {'ethernet': nic, 'ethernet_irq_affinity': irq,
                'ethernet_queues': static['queue_configuration'][interface]}
    signature = digest(selected)
    settings[signature] = selected
    run_config = ROOT / 'configs' / (label + '.json')
    runs[label] = {'job': job, 'complete': (ROOT / 'control' / (label + '-COMPLETE')).read_text().strip(),
                   'config_sha256': hashlib.sha256(run_config.read_bytes()).hexdigest(),
                   'nic_settings_sha256': signature, 'result_directory': str(run)}

rdma_mtu = {}
for block in re.split(r'(?m)^hca_id:\s*', static['rdma_devices']['stdout'])[1:]:
    device = block.splitlines()[0].strip()
    if device in ('mlx5_0', 'mlx5_4'):
        rdma_mtu[device] = re.search(r'active_mtu:\s*([^\n]+)', block)[1].strip()
ethernet_pci = re.search(r'bus-info:\s*(\S+)', nic['driver'])[1]
ethernet_numa = next(value['numa_node'] for value in static['rdma'].values()
                     if value['pci_device'].endswith('/' + ethernet_pci))
ucx_log = run_location('r1-velo-rdma')[1] / 'frontend-numa0.log'
with ucx_log.open() as f:
    ucx_version = next(line.strip() for line in f if re.search(r'Version [0-9]', line))

nodes = ROOT / 'manifests/campaign-original-nodes.txt'
if not nodes.exists():
    nodes = ROOT / 'manifests/nodes.txt'
out = {'nodes': nodes.read_text().splitlines(),
       'canonical_tcp_configuration': config, 'order': order, 'runs': runs,
       'frontend_nic_settings': settings, 'selected_rdma_active_mtu': rdma_mtu,
       'ethernet_numa_node': ethernet_numa, 'ucx_version_log_line': ucx_version,
       'hardware': {p.name: json.loads(p.read_text()) for p in (ROOT / 'manifests').glob('hardware-*.json')},
       'allocations': {p.name: p.read_text() for p in (ROOT / 'manifests').glob('allocation-*.txt')},
       'rustc': (ROOT / 'manifests/rustc.txt').read_text(),
       'packet_scope': 'All traffic on enP6p3s0f1np1 and selected mlx5_0:1 and mlx5_4:1 ports; response traffic is not isolated.',
       'memory_scope': 'Private anonymous resident pages sampled after drain; full NUMA maps and process status remain with each run.'}
path = ROOT / 'results/four-way/environment.json'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'nic_setting_variants': len(settings), 'rdma_active_mtu': rdma_mtu, 'runs': len(runs)}))
