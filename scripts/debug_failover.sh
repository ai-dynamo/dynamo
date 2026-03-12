#!/bin/bash
# Debug script for multinode failover GPU memory investigation
# Run from local machine with kubectl access

NS="multinode-failover"
LDR="vllm-mn-fo-0-worker-0-worker-ldr-fvnmx"
WKR="vllm-mn-fo-0-worker-0-worker-wkr-tjvht"

gpu_mem() {
    local pod=$1 container=$2 label=$3
    local mem=$(kubectl exec $pod -n $NS -c $container -- nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null)
    echo "  [$label] GPU: used=${mem%,*} MiB, free=${mem#*,} MiB"
}

gpu_procs() {
    local pod=$1 container=$2 label=$3
    echo "  [$label] GPU processes:"
    kubectl exec $pod -n $NS -c $container -- nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    echo ""
}

all_gpu_mem() {
    local label=$1
    echo "=== $label ==="
    gpu_mem $LDR engine-0 "leader-e0"
    gpu_mem $WKR engine-0 "worker-e0"
}

all_gpu_procs() {
    local label=$1
    echo "=== $label GPU processes ==="
    gpu_procs $LDR engine-0 "leader"
    gpu_procs $WKR engine-0 "worker"
}

kexec() {
    local pod=$1 container=$2
    shift 2
    kubectl exec $pod -n $NS -c $container -- "$@" 2>/dev/null
}

MODEL="Qwen/Qwen3-0.6B"
VLLM_CMD="python3 -m dynamo.vllm --model $MODEL --tensor-parallel-size 2 --distributed-executor-backend mp --load-format gms --gms-mode shadow"

echo "=============================================="
echo "  Multinode Failover GPU Memory Debug"
echo "=============================================="
echo ""

# ============================================================
# Experiment A: Baseline GPU memory
# ============================================================
echo "=== EXPERIMENT A: Baseline ==="
all_gpu_mem "Clean state (sleep infinity)"
all_gpu_procs "Clean state"

# ============================================================
# Experiment B: Start engine-0 group, measure memory at each stage
# ============================================================
echo ""
echo "=== EXPERIMENT B: Engine-0 lifecycle memory trace ==="

echo "Starting engine-0 leader..."
# Leader: node-rank 0, master-port 29500
kexec $LDR engine-0 bash -c "
ENGINE_ID=0 FAILOVER_LOCK_PATH=/shared/failover.lock DYN_SYSTEM_PORT=9090 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 DYN_VLLM_KV_EVENT_PORT=20080 \
SHADOW_SKIP_KV_CACHE=1 DYN_VLLM_GMS_MODE=shadow \
nohup $VLLM_CMD --nnodes 2 --node-rank 0 --master-addr \$(hostname) --master-port 29500 \
> /tmp/engine0_leader.log 2>&1 &
echo \$!
" &

echo "Waiting for leader TCP store..."
for i in $(seq 1 120); do
    if kexec $LDR engine-0 ss -tlnp 2>/dev/null | grep -q ":29500"; then
        echo "TCP store ready ($i)"
        break
    fi
    [ $i -eq 120 ] && echo "TIMEOUT"
    sleep 2
done

echo "Starting engine-0 worker (headless)..."
# Get leader hostname for worker
LEADER_HOST=$(kubectl exec $LDR -n $NS -c engine-0 -- hostname 2>/dev/null)
echo "Leader hostname: $LEADER_HOST"

kexec $WKR engine-0 bash -c "
SHADOW_SKIP_KV_CACHE=1 \
nohup python3 -m dynamo.vllm --model $MODEL --tensor-parallel-size 2 \
--distributed-executor-backend mp --load-format gms --headless \
--nnodes 2 --node-rank 1 --master-addr $LEADER_HOST --master-port 29500 \
> /tmp/engine0_worker.log 2>&1 &
echo \$!
" &

echo "Monitoring GPU memory during engine-0 init..."
for i in $(seq 1 60); do
    sleep 5
    all_gpu_mem "Engine-0 init T+${i}x5s"

    # Check if leader reached standby
    if kexec $LDR engine-0 grep -q "waiting for lock" /tmp/engine0_leader.log 2>/dev/null; then
        echo "*** Engine-0 reached STANDBY ***"
        break
    fi

    # Check for errors
    if kexec $LDR engine-0 grep -q "Error\|error\|OOM\|CUDA out" /tmp/engine0_leader.log 2>/dev/null; then
        echo "*** Engine-0 ERROR detected ***"
        kexec $LDR engine-0 tail -10 /tmp/engine0_leader.log
        break
    fi
done

all_gpu_mem "Engine-0 at STANDBY"
all_gpu_procs "Engine-0 at STANDBY"

# ============================================================
# Experiment C: Start engine-1 group, measure memory
# ============================================================
echo ""
echo "=== EXPERIMENT C: Engine-1 lifecycle ==="

echo "Starting engine-1 leader..."
kexec $LDR engine-1 bash -c "
ENGINE_ID=1 FAILOVER_LOCK_PATH=/shared/failover.lock DYN_SYSTEM_PORT=9091 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 DYN_VLLM_KV_EVENT_PORT=20081 \
SHADOW_SKIP_KV_CACHE=1 DYN_VLLM_GMS_MODE=shadow \
nohup $VLLM_CMD --nnodes 2 --node-rank 0 --master-addr \$(hostname) --master-port 29600 \
> /tmp/engine1_leader.log 2>&1 &
echo \$!
" &

echo "Waiting for engine-1 TCP store..."
for i in $(seq 1 120); do
    if kexec $LDR engine-1 ss -tlnp 2>/dev/null | grep -q ":29600"; then
        echo "TCP store ready ($i)"
        break
    fi
    [ $i -eq 120 ] && echo "TIMEOUT"
    sleep 2
done

echo "Starting engine-1 worker..."
kexec $WKR engine-1 bash -c "
SHADOW_SKIP_KV_CACHE=1 \
nohup python3 -m dynamo.vllm --model $MODEL --tensor-parallel-size 2 \
--distributed-executor-backend mp --load-format gms --headless \
--nnodes 2 --node-rank 1 --master-addr $LEADER_HOST --master-port 29600 \
> /tmp/engine1_worker.log 2>&1 &
echo \$!
" &

echo "Monitoring GPU memory during engine-1 init..."
for i in $(seq 1 60); do
    sleep 5
    all_gpu_mem "Engine-1 init T+${i}x5s"

    if kexec $LDR engine-1 grep -q "waiting for lock" /tmp/engine1_leader.log 2>/dev/null; then
        echo "*** Engine-1 reached STANDBY ***"
        break
    fi

    if kexec $LDR engine-1 grep -q "Error\|error\|OOM\|CUDA out" /tmp/engine1_leader.log 2>/dev/null; then
        echo "*** Engine-1 ERROR ***"
        kexec $LDR engine-1 tail -10 /tmp/engine1_leader.log
        break
    fi
done

all_gpu_mem "Both engines at STANDBY"
all_gpu_procs "Both engines at STANDBY"

# Check which engine won the lock
echo ""
echo "=== Lock status ==="
if kexec $LDR engine-0 grep -q "Lock acquired" /tmp/engine0_leader.log 2>/dev/null; then
    echo "Engine-0 is ACTIVE (won lock)"
    ACTIVE_E="engine-0"
    SHADOW_E="engine-1"
elif kexec $LDR engine-1 grep -q "Lock acquired" /tmp/engine1_leader.log 2>/dev/null; then
    echo "Engine-1 is ACTIVE (won lock)"
    ACTIVE_E="engine-1"
    SHADOW_E="engine-0"
else
    echo "Neither engine acquired lock yet..."
fi

echo ""
all_gpu_mem "After lock resolution"
all_gpu_procs "After lock resolution"

# Wait for active engine to finish waking
echo ""
echo "Waiting for active engine to register..."
for i in $(seq 1 60); do
    sleep 5
    if kexec $LDR $ACTIVE_E grep -q "Registered endpoint" /tmp/${ACTIVE_E/engine-/engine}_leader.log 2>/dev/null; then
        echo "Active engine registered ($i)"
        break
    fi
    all_gpu_mem "Active engine wake T+${i}x5s"
done

all_gpu_mem "FINAL: Active serving, shadow sleeping"
all_gpu_procs "FINAL: Active serving, shadow sleeping"

# ============================================================
# Experiment D: Kill active engine, track memory reclamation
# ============================================================
echo ""
echo "=== EXPERIMENT D: Kill active engine, track GPU memory ==="

echo "Killing $ACTIVE_E on leader pod..."
kexec $LDR $ACTIVE_E bash -c "kill -9 \$(cat /tmp/${ACTIVE_E/engine-/engine}_leader.pid 2>/dev/null) 2>/dev/null; pkill -9 -f 'master-port.*29${ACTIVE_E/engine-/}00' 2>/dev/null"

echo "Killing $ACTIVE_E on worker pod..."
kexec $WKR $ACTIVE_E bash -c "pkill -9 -f 'master-port.*29${ACTIVE_E/engine-/}00' 2>/dev/null; pkill -9 -f dynamo.vllm 2>/dev/null"

echo ""
echo "Tracking GPU memory reclamation (1s intervals)..."
for i in $(seq 1 30); do
    sleep 1
    L_MEM=$(kexec $LDR engine-0 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    W_MEM=$(kexec $WKR engine-0 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "  T+${i}s: leader=${L_MEM} MiB, worker=${W_MEM} MiB"
done

all_gpu_mem "After 30s reclamation"
all_gpu_procs "After 30s reclamation"

echo ""
echo "=== Shadow engine status ==="
kexec $LDR $SHADOW_E grep -E "Lock acquired|wake|Allocated KV|Registered|OOM|CUDA out" /tmp/${SHADOW_E/engine-/engine}_leader.log 2>/dev/null | tail -10
echo "---"
kexec $WKR $SHADOW_E grep -E "wake|Allocated KV|OOM|CUDA out" /tmp/${SHADOW_E/engine-/engine}_worker.log 2>/dev/null | tail -10

echo ""
echo "=============================================="
echo "  Debug complete"
echo "=============================================="
