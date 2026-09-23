"""Freeze the established workload and generate the balanced four-mode matrix."""
import argparse
import copy
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--dynamo', required=True)
p.add_argument('--velo', required=True)
p.add_argument('--rdma-device', required=True)
p.add_argument('--rdma-device-numa1', required=True)
a = p.parse_args()
base = json.loads((a.root / 'configs/template.json').read_text())
base['pins'].update(dynamo=a.dynamo, velo=a.velo, build_features='velo-ucx,tracing/release_max_level_warn')
base['network']['ucx_env'] = {'UCX_TLS': 'rc_mlx5,ud_mlx5,self', 'UCX_NET_DEVICES': a.rdma_device, 'UCX_LOG_LEVEL': 'info'}
base['network']['ucx_numa_devices'] = {'0': a.rdma_device, '1': a.rdma_device_numa1}
modes = ['tcp', 'quic', 'velo-tcp', 'velo-rdma']
order = [
    modes, ['tcp', 'quic', 'velo-rdma', 'velo-tcp'],
    ['quic', 'velo-tcp', 'tcp', 'velo-rdma'],
    ['velo-tcp', 'velo-rdma', 'quic', 'tcp'],
    ['velo-rdma', 'tcp', 'velo-tcp', 'quic'],
]
labels = []
for round_number, row in enumerate(order):
    for mode in row:
        label = f'{"discard" if round_number == 0 else "r" + str(round_number)}-{mode}'
        c = copy.deepcopy(base)
        c['campaign']['name'] = 'main-' + label
        c['runtime']['response_plane'] = 'velo' if mode.startswith('velo-') else mode
        c['runtime']['velo_response_transport'] = 'ucx' if mode == 'velo-rdma' else 'tcp'
        (a.root / 'configs' / (label + '.json')).write_text(json.dumps(c, indent=2) + '\n')
        labels.append(label)
(a.root / 'configs/order.json').write_text(json.dumps(labels, indent=2) + '\n')

for mode in ('velo-tcp', 'velo-rdma'):
    c = json.loads((a.root / 'configs' / ('discard-' + mode + '.json')).read_text())
    label = 'smoke-' + mode
    c['campaign'].update(name='main-' + label, smoke_only=True)
    c['runtime']['num_mockers'] = 12
    for node in c['topology']['mocker_nodes']:
        for process in node['processes']:
            process['workers'] = 1
    (a.root / 'configs' / (label + '.json')).write_text(json.dumps(c, indent=2) + '\n')
c = json.loads((a.root / 'configs/discard-velo-rdma.json').read_text())
c['campaign'].update(name='main-preflight-velo-rdma', fixed_concurrency=8192, preflight_saturation=True)
c['saturation']['initial_candidates'] = [8192]
c['runtime']['mocker_system_metrics_base_port'] = 9100
(a.root / 'configs/preflight-velo-rdma.json').write_text(json.dumps(c, indent=2) + '\n')
