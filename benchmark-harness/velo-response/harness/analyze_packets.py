"""Use common measurement boundaries for hardware, host and CPU counters."""
import json
import sys
from pathlib import Path
from analyze_campaign import ROOT, rows, epoch, scalar


def flattened(value, prefix=''):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(flattened(child, prefix + '/' + str(key)))
    elif isinstance(value, (float, int)):
        out[prefix] = value
    return out


def analyze(label):
    job = (ROOT / 'control/job-id').read_text().strip()
    run = ROOT / 'results' / f'{job}-main-{label}' / 'ablations' / f'main-{label}'
    path = run / 'matching-window-results.json'
    result = json.loads(path.read_text())
    start, end = (result['window'][k] for k in ('epoch_start', 'epoch_end'))
    duration = end - start
    requests = result['headline']['requests']
    tokens = result['headline']['output_tokens']
    network = {}
    for rank in range(5):
        samples = []
        clock_ticks = None
        source = ROOT / 'artifacts' / f'network-{label}' / f'rank-{rank}' / 'samples.jsonl'
        for row in rows(source):
            clock_ticks = row['clock_ticks_per_second']
            t = epoch(row['timestamp'])
            if t < start - 10:
                continue
            values = flattened({k: row[k] for k in ('network', 'ethtool', 'rdma', 'snmp', 'netstat')})
            cpu = row['proc_stat']['cpu']
            values.update(system_ticks=cpu[2], softirq_ticks=cpu[6], irq_ticks=cpu[5])
            samples.append({'time': t, 'values': values})
            if t > end + 2:
                break
        if len(samples) < 2:
            raise ValueError(f'{source}: insufficient packet telemetry')
        keys = set.intersection(*(set(row['values']) for row in samples))
        delta = {key: scalar(samples, end, lambda r: r['values'][key]) - scalar(samples, start, lambda r: r['values'][key]) for key in keys}
        hardware = {key: value for key, value in delta.items() if key.startswith(('/ethtool/', '/rdma/')) and ('packet' in key or 'pkts' in key or 'port_rcv_data' in key or 'port_xmit_data' in key)}
        wire = {key: value for key, value in delta.items() if key.endswith(('/rx_packets_phy', '/tx_packets_phy', '/counters/port_rcv_packets', '/counters/port_xmit_packets'))}
        if not wire or any(value < 0 for value in wire.values()):
            raise ValueError(f'{source}: missing or reset hardware packet counters')
        network[str(rank)] = {
            'wire_rx_size_bucket_deltas': {key: value for key, value in delta.items() if key.startswith('/ethtool/') and key.endswith('_bytes_phy') and '/rx_' in key},
            'wire_packets_per_second': {key: value / duration for key, value in wire.items()},
            'wire_packets_per_completed_request': {key: value / requests for key, value in wire.items()} if rank == 0 else None,
            'wire_packets_per_output_token': {key: value / tokens for key, value in wire.items()} if rank == 0 else None,
            'seconds': duration, 'delta': delta,
            'device_counter_rates': {key: value / duration for key, value in hardware.items()},
            'device_counters_per_completed_request': {key: value / requests for key, value in hardware.items()} if rank == 0 else None,
            'device_counters_per_output_token': {key: value / tokens for key, value in hardware.items()} if rank == 0 else None,
            'system_cpu_seconds': delta['system_ticks'] / clock_ticks,
            'softirq_cpu_seconds': delta['softirq_ticks'] / clock_ticks,
            'irq_cpu_seconds': delta['irq_ticks'] / clock_ticks,
            'scope': 'whole node; Ethernet and RDMA ports are separate; host packet counters include offload aggregation',
        }
    result['network'] = network
    result['packet_normalization'] = 'Frontend completion and output-token counters over the same interpolated measurement window. IB data counters use four-byte units. Hardware packet buckets, when present, are retained without converting host packet sizes to wire sizes.'
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    for label in sys.argv[1:]:
        analyze(label)
