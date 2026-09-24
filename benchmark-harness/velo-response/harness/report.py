"""Summarize four balanced repeats; retain per-run values and baseline ratios."""
import csv
import json
from pathlib import Path
import re
import statistics
from analyze_campaign import ROOT, run_location

MODES = ('tcp', 'quic', 'velo-tcp', 'velo-rdma')

def measurements(result):
    h = result['headline']
    n = result['network']['0']
    d = n['delta']
    seconds = result['window']['seconds']
    out = {k: h[k] for k in ('requests_per_second', 'output_tokens_per_second',
           'frontend_cpu_ms_per_request', 'frontend_cpu_us_per_output_token',
           'frontend_mean_cores')}
    out.update(frontend_system_cores=h['frontend_system_cpu_seconds']/seconds,
               frontend_ucx_progress_cores=h['frontend_ucx_progress_cpu_seconds']/seconds,
               host_system_cores=n['system_cpu_seconds']/seconds,
               host_softirq_cores=n['softirq_cpu_seconds']/seconds)
    for direction in ('rx', 'tx'):
        out[f'ethernet_{direction}_host_packets_per_second'] = d[
            f'/network/enP6p3s0f1np1/{direction}_packets'] / seconds
    for metric in ('time_to_first_token', 'inter_token_latency', 'request_latency'):
        value = result['client']['distributions'][metric]['distribution']
        for percentile in ('p50', 'p95', 'p99'):
            out[f'{metric}_{percentile}_ms'] = value[percentile]
    for direction, ethernet, rdma in (('rx', 'rx_packets_phy', 'port_rcv_packets'),
                                      ('tx', 'tx_packets_phy', 'port_xmit_packets')):
        wire_rates = n['wire_packets_per_second']
        eth = seconds * sum(v for k, v in wire_rates.items() if k.endswith('/' + ethernet))
        ib = seconds * sum(v for k, v in wire_rates.items() if k.endswith('/counters/' + rdma))
        for fabric, packets in (('ethernet', eth), ('rdma', ib), ('total', eth + ib)):
            out[f'{fabric}_{direction}_wire_packets_per_second'] = packets / seconds
            out[f'{fabric}_{direction}_wire_packets_per_request'] = packets / h['requests']
            out[f'{fabric}_{direction}_wire_packets_per_output_token'] = packets / h['output_tokens']
    for unit in ('second', 'request', 'output_token'):
        out[f'total_wire_packets_per_{unit}'] = sum(
            out[f'total_{direction}_wire_packets_per_{unit}'] for direction in ('rx', 'tx'))
    eth_rx = sum(v for k, v in d.items() if k.endswith('/rx_packets_phy'))
    eth_rx_bytes = sum(v for k, v in d.items() if k.endswith('/rx_bytes_phy'))
    out['ethernet_mean_rx_wire_bytes'] = eth_rx_bytes / eth_rx
    for key, packets in n['wire_rx_size_bucket_deltas'].items():
        bucket = key.rsplit('/', 1)[-1]
        if re.fullmatch(r'rx_\d+(?:_to_\d+)?_bytes_phy', bucket):
            out['ethernet_' + bucket + '_percent'] = 100 * packets / eth_rx
    for counter in ('rx_discards_phy', 'tx_discards_phy', 'rx_out_of_buffer', 'RetransSegs'):
        out[counter + '_per_second'] = sum(v for k, v in d.items() if k.endswith('/' + counter)) / seconds
    rdma_prefixes = tuple({key.split('/counters/', 1)[0] + '/'
                           for key in n['wire_packets_per_second'] if key.startswith('/rdma/')})
    for counter in ('port_rcv_errors', 'port_xmit_discards', 'out_of_buffer',
                    'packet_seq_err', 'local_ack_timeout_err', 'req_transport_retries_exceeded'):
        out['rdma_' + counter + '_per_second'] = sum(
            value for key, value in d.items()
            if key.startswith(rdma_prefixes) and key.endswith('/' + counter)) / seconds
    fe = result['telemetry']['frontend-system-telemetry.jsonl']['processes']
    out['frontend_mean_rss_gib'] = sum(v['rss_bytes']['mean'] for k, v in fe.items() if k.startswith('frontend')) / 2**30
    client = result['telemetry'].get('aiperf-system-telemetry.jsonl', {}).get('processes', {})
    out['client_mean_cores'] = sum(v['mean_cores'] for v in client.values())
    return out

runs = {}
for mode in MODES:
    runs[mode] = []
    for repeat in range(1, 5):
        label = f'r{repeat}-{mode}'
        job, run = run_location(label)
        result = json.loads((run / 'matching-window-results.json').read_text())
        assert result['quality']['accepted'], label
        client = json.loads((run / 'campaign-result.json').read_text())['aiperf']
        residency = {str(node): json.loads((run / f'numa{node}-settled-residency.json').read_text())
                     for node in (0, 1)}
        runs[mode].append({'label': label, 'job': job, 'metrics': measurements(result),
                           'window': result['window'],
                           'active_http_requests': result['actual_active_http_requests'],
                           'quality': result['quality'],
                           'client_counts': result['client']['counts'],
                           'client_health': {key: client.get(key) for key in
                               ('client_health_accepted', 'event_loop_warning_count')},
                           'frontend_private_anonymous_pages_by_node': residency,
                           'error_examples': result['client']['error_examples']})

summary = {}
for mode, records in runs.items():
    summary[mode] = {}
    for name in records[0]['metrics']:
        values = [r['metrics'][name] for r in records]
        summary[mode][name] = {'median': statistics.median(values), 'min': min(values),
                              'max': max(values), 'values': values}
for mode in MODES:
    for name, metric in summary[mode].items():
        for baseline in ('tcp', 'quic'):
            denominator = summary[baseline][name]['median']
            metric['percent_vs_' + baseline] = 100 * (metric['median'] / denominator - 1) if denominator else None

qualification = {}
for mode in MODES:
    checks = {}
    for name in ('requests_per_second', 'output_tokens_per_second'):
        checks[name] = summary[mode][name]['percent_vs_tcp'] >= -5
    for name in ('time_to_first_token_p99_ms', 'inter_token_latency_p99_ms', 'request_latency_p99_ms'):
        checks[name] = summary[mode][name]['percent_vs_tcp'] <= 5
    checks['frontend_cpu_ms_per_request'] = summary[mode]['frontend_cpu_ms_per_request']['percent_vs_tcp'] <= 10
    qualification[mode] = {'checks': checks, 'passes_provisional_limits': all(checks.values())}

out = ROOT / 'results/four-way'
out.mkdir(exist_ok=True)
(out / 'summary.json').write_text(json.dumps({'frozen_build': json.loads(
    (ROOT / 'manifests/frozen-build.json').read_text()), 'runs': runs, 'summary': summary,
    'qualification': qualification}, indent=2) + '\n')
with (out / 'all-metrics.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['mode', 'metric', 'median', 'min', 'max', 'percent_vs_tcp', 'percent_vs_quic'])
    for mode in MODES:
        for name, m in summary[mode].items():
            writer.writerow([mode, name, m['median'], m['min'], m['max'], m['percent_vs_tcp'], m['percent_vs_quic']])
lines = ['# Tyche Velo response comparison', '',
         'Values are medians of four balanced runs. The CSV includes each range and changes against both TCP and QUIC.', '',
         '| Metric | TCP | QUIC | Velo TCP | Velo RDMA |', '|---|---:|---:|---:|---:|']
for name in summary['tcp']:
    lines.append('| ' + name + ' | ' + ' | '.join(f'{summary[m][name]["median"]:.3f}' for m in MODES) + ' |')
lines += ['', 'Provisional limits against TCP:', '']
for mode in MODES:
    failed = [name for name, passed in qualification[mode]['checks'].items() if not passed]
    lines.append(f'- {mode}: ' + ('pass' if not failed else 'fail: ' + ', '.join(failed)))
lines += ['', 'Small error counts are retained in this directional comparison.', '',
          '| Run | Exported completed records | Errors | Error fraction | Cancelled at drain deadline | Client event-loop warnings |',
          '|---|---:|---:|---:|---:|---:|']
for mode in MODES:
    for record in runs[mode]:
        q = record['quality']
        lines.append(f'| {record["label"]} | {record["client_counts"]["profiling_records"]} | '
                     f'{q["error_count"]} | {100*q["error_fraction"]:.6f}% | '
                     f'{q.get("phase_cancelled_requests",0)} | '
                     f'{record["client_health"]["event_loop_warning_count"]} |')
lines += ['', 'The Ethernet and RDMA fabrics differ. Hardware counters cover all traffic on the measured frontend ports. '
          'Totals sum the Ethernet interface and the two selected InfiniBand ports. '
          'Host packet aggregation and RDMA completions are not wire packets. '
          'Latency percentiles describe successful exported requests; requests cancelled at the drain deadline are counted separately and limit tail comparisons. '
          'Packet changes are reported separately from performance qualification. Defaults are unchanged.', '']
(out / 'metrics.md').write_text('\n'.join(lines))
print(out)
