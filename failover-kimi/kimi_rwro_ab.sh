#!/bin/bash
# LAYER 0 + numerical A/B — GMS RW->RO in isolation, then compare RW vs RO
# logprobs across several diverse-margin prompts (not just greedy argmax on one
# high-confidence prompt). Confirms the RO MoE-kernel rebuild is numerically
# equivalent, not merely argmax-equivalent. Kimi-K2.6 TP8.
set -u
source /opt/dynamo/venv/bin/activate 2>/dev/null || true
export HF_HOME=/tmp/hf HF_HUB_OFFLINE=0; mkdir -p /tmp/hf
MODEL=/tmp/kimi-k2.6-nvfp4; SERVED=kimi-k2.6; TP=8; MML=4096; UTIL=0.8
NOAT=${DYN_NO_AUTOTUNE:-1}; EAGER=${EAGER:-1}
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
OUT=/tmp/kimi_rwro_ab; rm -rf "$OUT"; mkdir -p "$OUT/logs"
have(){ grep -aq "$2" "$1" 2>/dev/null; }
log(){ echo "[rwroab $(date +%T)] $*"; }
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -"$TP" | tr '\n' ' '; echo; }
cleanup(){ pkill -9 -f "[d]ynamo.vllm" 2>/dev/null; pkill -9 -f "[d]ynamo.frontend" 2>/dev/null
  pkill -9 -f "[E]ngineCore" 2>/dev/null; pkill -9 -f "[g]pu_memory_service" 2>/dev/null; sleep 2
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  rm -f /tmp/gms_*.sock; cp "$OUT"/logs/*.log "$OUT/" 2>/dev/null; }
trap cleanup EXIT
cleanup; sleep 4

# diverse-margin prompts: high-confidence factual -> open-ended (lower margin)
PROMPTS=(
  "The capital of France is"
  "The chemical symbol for gold is"
  "Two plus two equals"
  "My favorite season of the year is"
)
# JSON-safe POST: phase idx prompt -> $OUT/${phase}_${idx}.json ; echo http code
infer(){ local phase=$1 idx=$2 p=$3
  python3 - "$SERVED" "$p" "$OUT/${phase}_${idx}.json" <<'PY'
import json,sys,urllib.request
served,prompt,outfile=sys.argv[1],sys.argv[2],sys.argv[3]
body=json.dumps({"model":served,"prompt":prompt,"max_tokens":10,"temperature":0,"logprobs":5}).encode()
req=urllib.request.Request("http://localhost:8000/v1/completions",data=body,headers={"Content-Type":"application/json"})
try:
    r=urllib.request.urlopen(req,timeout=60); d=json.load(r); open(outfile,"w").write(json.dumps(d)); print(200)
except Exception as e:
    open(outfile,"w").write(json.dumps({"error":str(e)})); print("ERR")
PY
}
run_all(){ local phase=$1 i=0
  for p in "${PROMPTS[@]}"; do i=$((i+1)); local code; code=$(infer "$phase" "$i" "$p")
    log "  $phase[$i] HTTP $code :: $(python3 -c "import json;d=json.load(open('$OUT/${phase}_${i}.json'));print(''.join(d['choices'][0]['logprobs']['tokens'])[:60])" 2>/dev/null)"
  done; }

launch(){ DYN_NO_AUTOTUNE=$NOAT DYN_SYSTEM_PORT=$((8100+$1)) \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600+$1)) DYN_VLLM_KV_EVENT_PORT=$((20080+$1)) \
  nohup python3 -m dynamo.vllm --model "$MODEL" --served-model-name "$SERVED" -tp "$TP" \
  --trust-remote-code --max-model-len "$MML" --gpu-memory-utilization "$UTIL" $EAGER_FLAG \
  --load-format gms > "$2" 2>&1 & echo $!; }

log "GMS weights servers (TP=$TP)"
for d in $(seq 0 $((TP-1))); do python3 -m gpu_memory_service --device $d --tag weights > "$OUT/logs/gms_w$d.log" 2>&1 & done
for d in $(seq 0 $((TP-1))); do for i in $(seq 1 90); do have "$OUT/logs/gms_w$d.log" "Server started" && break; sleep 1; done; done
nohup python3 -m dynamo.frontend > "$OUT/logs/frontend.log" 2>&1 &

log "=== A) RW engine: disk -> GMS -> commit -> serve ==="
A=$(launch 0 "$OUT/logs/engineA.log"); log "engineA pid $A"
for i in $(seq 1 700); do have "$OUT/logs/engineA.log" "Registered endpoint" && { log "RW registered +${i}s"; break; }
  kill -0 $A 2>/dev/null || { log "engineA DIED"; grep -anE "Error|Traceback|Cannot copy|meta tensor|not validated" "$OUT/logs/engineA.log" | tail -10; exit 1; }; sleep 1; done
have "$OUT/logs/engineA.log" "Registered endpoint" || { log "RW never registered"; exit 1; }
for i in $(seq 1 60); do have "$OUT/logs/frontend.log" "Completions is ready" && break; sleep 1; done
log "RW mem: $(mem)"; run_all rw

log "=== kill A (committed weights must survive in GMS servers) ==="
pkill -9 -P "$A" 2>/dev/null; kill -9 "$A" 2>/dev/null; sleep 8
log "post-kill mem: $(mem)"

log "=== B) RO engine: import committed weights + rebuild MoE kernel -> serve ==="
B=$(launch 1 "$OUT/logs/engineB.log"); log "engineB pid $B"
for i in $(seq 1 700); do have "$OUT/logs/engineB.log" "Registered endpoint" && { log "RO registered +${i}s"; break; }
  kill -0 $B 2>/dev/null || { log "engineB DIED"; grep -anE "Error|Traceback|Cannot copy|meta tensor|not validated|moe_kernel" "$OUT/logs/engineB.log" | tail -12; exit 1; }; sleep 1; done
have "$OUT/logs/engineB.log" "Registered endpoint" || { log "RO never registered"; exit 1; }
sleep 3
log "RO mem: $(mem)"; run_all ro
grep -a "rebuilt .* MoE kernel\|NVFP4 backend" "$OUT/logs/engineB.log" | head -2

log "=== NUMERICAL A/B: RW vs RO logprobs per prompt ==="
python3 - "$OUT" "${#PROMPTS[@]}" <<'PY'
import json,sys
out,n=sys.argv[1],int(sys.argv[2])
worst=0.0; anymismatch=False
for i in range(1,n+1):
    try:
        rw=json.load(open(f"{out}/rw_{i}.json")); ro=json.load(open(f"{out}/ro_{i}.json"))
        rwl=rw["choices"][0]["logprobs"]; rol=ro["choices"][0]["logprobs"]
        rwt,rot=rwl["tokens"],rol["tokens"]; rwp,rop=rwl["token_logprobs"],rol["token_logprobs"]
        tok_match=(rwt==rot)
        m=max((abs((a or 0)-(b or 0)) for a,b in zip(rwp,rop)), default=0.0)
        worst=max(worst,m); anymismatch=anymismatch or (not tok_match)
        print(f"  prompt {i}: tokens_match={tok_match}  max|dlogprob|={m:.2e}  RW='{''.join(rwt)[:40]}'")
        if not tok_match:
            print(f"     RW tokens: {rwt}")
            print(f"     RO tokens: {rot}")
    except Exception as e:
        print(f"  prompt {i}: COMPARE FAILED {e}")
verdict = "PASS" if (not anymismatch and worst < 1e-2) else "REVIEW"
print(f"AB_VERDICT={verdict} worst_max_abs_dlogprob={worst:.3e} token_mismatch={anymismatch}")
PY
echo "KIMI_RWRO_AB_DONE"
