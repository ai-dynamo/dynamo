"""Analyze counters and CPU in common windows; keep full client records on disk."""
import bisect, collections, datetime as dt, json, math, re, statistics, sys
from pathlib import Path
ROOT=Path('/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923')
def epoch(x): return dt.datetime.fromisoformat(x).timestamp()
def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def dist(v):
    v=sorted(v)
    if not v:return None
    def q(p):
        i=(len(v)-1)*p;a=int(i);return v[a]+(v[min(a+1,len(v)-1)]-v[a])*(i-a)
    return dict(n=len(v),mean=statistics.fmean(v),p50=q(.5),p95=q(.95),p99=q(.99),min=v[0],max=v[-1])
def prom(text):
    out={}
    for line in text.splitlines():
        if line.startswith('#'):continue
        m=re.match(r'([^ {]+)(\{.*\})?\s+([-+.0-9eE]+)',line)
        if m:out[m[1]+(m[2] or '')]=float(m[3])
    return out
def total(m,name):return sum(v for k,v in m.items() if k.split('{')[0]==name)
def rdma_lanes(run,config):
    if config['runtime']['response_plane']!='velo' or config['runtime']['velo_response_transport']!='ucx':
        return None
    devices=config['network']['ucx_numa_devices']
    logs={f'frontend-numa{node}.log':device for node,device in devices.items()}
    for node in config['topology']['mocker_nodes']:
        for process in node['processes']:
            logs[process['name']+'.log']=devices[process['memory'].removeprefix('bind:')]
    evidence={}
    for name,device in logs.items():
        lanes=[]
        with (run/name).open(errors='replace') as f:
            for line in f:
                lanes.extend(re.findall(r'\bam\(([^)]*)\)',line))
        verified=bool(lanes) and all('rc_mlx5/'+device in lane and 'tcp/' not in lane for lane in lanes)
        evidence[name]={'expected_device':device,'active_message_lanes':sorted(set(lanes)),'verified':verified}
    return evidence
def read_telemetry(path,start=None,end=None):
    out=[]
    for r in rows(path):
        t=epoch(r['timestamp'])
        if start is not None and t<start-3:continue
        if end is not None and t>end+3:break
        x={'time':t,'monotonic':r['monotonic_seconds'], 'metrics':prom((r.get('metrics') or {}).get('text','')),'processes':{}}
        for name,procs in r.get('tracked_processes',{}).items():
            x['processes'][name]={
                'user_seconds':sum(p['user_ticks'] for p in procs)/r['clock_ticks_per_second'],
                'system_seconds':sum(p['system_ticks'] for p in procs)/r['clock_ticks_per_second'],
                'rss_bytes':sum(p['rss_pages'] for p in procs)*r['page_size'],
                'threads':sum(p['thread_count'] for p in procs),
                'ucx_progress_seconds':sum(t['user_ticks']+t['system_ticks'] for group in r.get('tracked_threads',{}).get(name,[]) for t in group['threads'] if t['comm'].startswith('velo-ucx'))/r['clock_ticks_per_second']}
        out.append(x)
    return out
def bracket(data,t):
    i=bisect.bisect_left([r['time'] for r in data],t)
    if i==0 or i==len(data):raise ValueError(('window outside telemetry',t,data[0]['time'],data[-1]['time']))
    a,b=data[i-1:i+1]; w=(t-a['time'])/(b['time']-a['time']);return a,b,w

def scalar(data,t,get):
    a,b,w=bracket(data,t);return get(a)+(get(b)-get(a))*w

def window(data,start,end):
    duration=end-start
    metric_keys=set().union(*(r['metrics'].keys() for r in data))
    md={k:scalar(data,end,lambda r:r['metrics'].get(k,0))-scalar(data,start,lambda r:r['metrics'].get(k,0)) for k in metric_keys}
    inside=[r for r in data if start<=r['time']<=end]
    out={'seconds':duration,'metrics_delta':md,'processes':{},'interpolation':'linear between adjacent one-second telemetry samples','max_sample_gap_seconds':max(b['time']-a['time'] for a,b in zip(data,data[1:]))}
    for name in data[0]['processes']:
        get=lambda key: scalar(data,end,lambda r:r['processes'].get(name,{}).get(key,0))-scalar(data,start,lambda r:r['processes'].get(name,{}).get(key,0))
        u,s=get('user_seconds'),get('system_seconds')
        out['processes'][name]=dict(user_cpu_seconds=u,system_cpu_seconds=s,cpu_seconds=u+s,mean_cores=(u+s)/duration,
            ucx_progress_cpu_seconds=get('ucx_progress_seconds'),rss_bytes=dist([r['processes'].get(name,{}).get('rss_bytes',0) for r in inside]),threads=dist([r['processes'].get(name,{}).get('threads',0) for r in inside]))
    out['active_http_requests']=dist([total(r['metrics'],'dynamo_frontend_active_requests') for r in inside])
    out['kv_sources']=dist([sum(v for k,v in r['metrics'].items() if k.startswith('dynamo_component_router_kv_zmq_ingress_sources{') and 'state="active"' in k) for r in inside])
    return out

def analyze(label):
    job=(ROOT/'control/job-id').read_text().strip()
    run=ROOT/'results'/(job+'-main-'+label)/'ablations'/('main-'+label)
    end_source='MEASUREMENT_ENDED' if (run/'MEASUREMENT_ENDED').exists() else 'SENDING_ENDED'
    start=epoch((run/'MEASUREMENT_STARTED').read_text().strip());end=epoch((run/end_source).read_text().strip())
    duration=end-start
    assert 118<=duration<=122,duration
    datasets={p.name:read_telemetry(p,start,end) for p in run.glob('*system-telemetry.jsonl')}
    config=json.loads((ROOT/'configs'/(label+'.json')).read_text())
    result={'label':label,'window':{'epoch_start':start,'epoch_end':end,'seconds':duration,'nominal_seconds':120,'end_source':end_source},'trace_concurrency':config['campaign']['fixed_concurrency'],'telemetry':{n:window(d,start,end) for n,d in datasets.items()}}
    names=['frontend-system-telemetry.jsonl','frontend-numa1-system-telemetry.jsonl']
    metrics=collections.Counter()
    for name in names:
        for k,v in result['telemetry'][name]['metrics_delta'].items():metrics[k]+=v
    req=total(metrics,'dynamo_frontend_requests_total');tok=total(metrics,'dynamo_frontend_output_tokens_total')
    process=result['telemetry'][names[0]]['processes']
    fe={k:v for k,v in process.items() if k.startswith('frontend')}
    cpu=sum(v['cpu_seconds'] for v in fe.values())
    for index,name in enumerate(names):
        own=result['telemetry'][name]['metrics_delta']
        own_req=total(own,'dynamo_frontend_requests_total');own_tok=total(own,'dynamo_frontend_output_tokens_total')
        own_cpu=result['telemetry'][name]['processes']['frontend-numa'+str(index)]
        own_cpu['cpu_ms_per_own_request']=own_cpu['cpu_seconds']*1000/own_req if own_req else None
        own_cpu['cpu_us_per_own_output_token']=own_cpu['cpu_seconds']*1e6/own_tok if own_tok else None
    for name,t in result['telemetry'].items():
        for pname,p in t['processes'].items():
            p['cpu_ms_per_global_request']=p['cpu_seconds']*1000/req if req else None
            p['cpu_us_per_global_output_token']=p['cpu_seconds']*1e6/tok if tok else None
    cached=total(metrics,'dynamo_frontend_tokenizer_cache_cached_tokens_total');uncached=total(metrics,'dynamo_frontend_tokenizer_cache_uncached_tokens_total')
    hits=total(metrics,'dynamo_frontend_tokenizer_cache_hits_total');misses=total(metrics,'dynamo_frontend_tokenizer_cache_misses_total')
    result['headline']={'requests':req,'output_tokens':tok,'requests_per_second':req/duration,'output_tokens_per_second':tok/duration,
        'frontend_cpu_seconds':cpu,'frontend_mean_cores':cpu/duration,'frontend_cpu_ms_per_request':cpu*1000/req if req else None,'frontend_cpu_us_per_output_token':cpu*1e6/tok if tok else None,
        'frontend_ucx_progress_cpu_seconds':sum(v['ucx_progress_cpu_seconds'] for v in fe.values()),'frontend_user_cpu_seconds':sum(v['user_cpu_seconds'] for v in fe.values()),'frontend_system_cpu_seconds':sum(v['system_cpu_seconds'] for v in fe.values()),
        'cache_hits':hits,'cache_misses':misses,'cache_hit_fraction':hits/(hits+misses) if hits+misses else None,'cached_input_tokens':cached,'uncached_input_tokens':uncached,'cached_token_fraction':cached/(cached+uncached) if cached+uncached else None}
    # Re-sample both active gauges on a common clock before summing.
    active=[]
    for i in range(int(duration)):
        t=start+i+.5
        active.append(sum(scalar(datasets[n],t,lambda r:total(r['metrics'],'dynamo_frontend_active_requests')) for n in names))
    result['actual_active_http_requests']=dist(active)
    result['frontend_counters']=dict(sorted(metrics.items()))
    result['client']=client_summary(run,start,end)
    campaign=json.loads((run/'campaign-result.json').read_text())
    counts=result['client']['counts']
    counts['phase_cancelled_requests']=campaign['aiperf'].get('phase_cancelled_requests',0)
    bad_counts={k:v for k,v in counts.items() if v and any(word in k for word in ('cancellations','mismatches','missing_or_duplicate','skips'))}
    error_count=counts.get('measured_errors',0)+counts.get('outside_window_errors',0)
    error_fraction=error_count/counts['profiling_records']
    expected=config['runtime']['num_mockers']
    result['rdma_lanes']=rdma_lanes(run,config)
    result['quality']={
        'complete':(run/'COMPLETE').exists(),
        'harness_accepted':campaign['accepted'],
        'invalid_client_records':bad_counts,
        'error_count':error_count,
        'error_fraction':error_fraction,
        'phase_cancelled_requests':counts['phase_cancelled_requests'],
        'small_error_count_accepted':error_fraction<=config['workload']['allowed_error_fraction'],
        'kv_sources_complete':all(result['telemetry'][n]['kv_sources']['min']==expected for n in names),
        'rdma_lanes_verified':result['rdma_lanes'] is None or all(x['verified'] for x in result['rdma_lanes'].values()),
    }
    result['quality']['accepted']=all((result['quality']['complete'],result['quality']['harness_accepted'],not bad_counts,result['quality']['kv_sources_complete'],result['quality']['small_error_count_accepted'],result['quality']['rdma_lanes_verified']))
    if (run/'profile-window-start.json').exists():
        p=json.loads((run/'profile-window-start.json').read_text());a=p['epoch'];b=a+15
        result['profile_launch_window']={'start':a,'end':b,'clock':'monotonic','note':'Controller launch window; actual perf sample windows are in frontend-profile-summary.json.','telemetry':{n:window(d,a,b) for n,d in datasets.items()}}
    (run/'matching-window-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'label':label,'headline':result['headline'],'active_http':result['actual_active_http_requests'],'client':result['client']},indent=2))
    return result

def client_summary(run,start,end):
    stats=collections.defaultdict(list);counts=collections.Counter();unit={};examples=[];request_ids=set()
    path=run/'load_artifacts/profile_export.jsonl'
    for r in rows(path):
        m=r.get('metadata',{}); counts['all_exported_records']+=1
        if m.get('benchmark_phase') not in (None,'profiling'):continue
        counts['profiling_records']+=1
        rid=m.get('x_request_id')
        if not rid or rid in request_ids:counts['missing_or_duplicate_request_ids']+=1
        request_ids.add(rid)
        if m.get('context_overflow_skip'):counts['context_overflow_skips']+=1
        en=m.get('request_end_ns',0)/1e9;st=m.get('request_start_ns',0)/1e9
        inwindow=start<=en<end
        metrics=r.get('metrics',{})
        def val(k):
            x=metrics.get(k);return x.get('value') if isinstance(x,dict) else x
        if m.get('was_cancelled'):
            counts['measured_cancellations' if inwindow else 'outside_window_cancellations']+=1
        if m.get('error') or m.get('error_code') or r.get('error'):
            counts['measured_errors' if inwindow else 'outside_window_errors']+=1
            if len(examples)<5:examples.append(m)
        output,usage=val('output_token_count'),val('usage_completion_tokens')
        if output is not None and usage is not None and output!=usage:counts['terminal_token_mismatches']+=1
        if inwindow:counts['measurement_completions']+=1
        if not start<=st<end:continue
        counts['measurement_arrivals']+=1
        if m.get('was_cancelled') or m.get('error') or m.get('error_code') or r.get('error'):continue
        counts['measurement_arrivals_completed']+=1
        for key in ['request_latency','time_to_first_token','time_to_first_output_token','inter_token_latency','input_token_count','input_sequence_length','output_sequence_length','output_token_count','usage_prompt_tokens','usage_completion_tokens']:
            v=val(key)
            if isinstance(v,(int,float)) and math.isfinite(v):
                stats[key].append(v);unit[key]=metrics[key].get('unit') if isinstance(metrics[key],dict) else None
        if val('inter_token_latency') is None and (usage or output or 0)>1:
            latency=val('request_latency');ttft=val('time_to_first_token') or val('time_to_first_output_token')
            if latency is not None and ttft is not None: stats['derived_mean_itl_ms'].append((latency-ttft)/((usage or output)-1))
    return {'cohort':'latencies for successful requests started in the observed measurement window, including drain completions; whole-export correctness checked','counts':dict(counts),'distributions':{k:dict(distribution=dist(v),unit=unit.get(k,'ms')) for k,v in stats.items()},'error_examples':examples}

if __name__=='__main__':
    results=[analyze(label) for label in sys.argv[1:]]
    (ROOT/'results'/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
