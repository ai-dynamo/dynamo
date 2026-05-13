#!/usr/bin/env python3
"""Generate dynamo-kvrouter bench scripts for Prefill {1..4}xTP1 with Decode {1,2,3}xTP2 or 1xTP4.

Each generated script mirrors run_benchx_1ctx4gen_dynamo_kvrouter.sh, but:
  - decode workers use TP2 (gen_config_tp2.yaml) or TP4 (gen_config_tp4.yaml)
  - prefill workers stay TP1 on NODE0
  - 3xTP2 decode (6 GPUs) overflows to NODE2, so those scripts request 3 nodes
"""

from pathlib import Path

OUT_DIR = Path("/home/rihuo/workspace/dynamo")
TEMPLATE_PATH = OUT_DIR / "run_benchx_1ctx4gen_dynamo_kvrouter.sh"


def decode_placement(dc: int, dtp: int):
    """Return list of (instance_idx, node_var, cuda_devices, dyn_port) tuples."""
    out = []
    if dtp == 4:
        out.append((0, "$NODE1", "0,1,2,3", 8085))
    elif dtp == 2:
        slots_node1 = [("$NODE1", "0,1"), ("$NODE1", "2,3")]
        slots_node2 = [("$NODE2", "0,1"), ("$NODE2", "2,3")]
        slots = slots_node1 + slots_node2
        for i in range(dc):
            node_var, gpus = slots[i]
            out.append((i, node_var, gpus, 8085 + i))
    return out


def render(p: int, dc: int, dtp: int) -> str:
    needs_node2 = dtp == 2 and dc == 3
    nnodes = 3 if needs_node2 else 2
    decodes = decode_placement(dc, dtp)
    gen_cfg = f"gen_config_tp{dtp}.yaml"
    suffix = f"{p}ctx{dc}gen_tp{dtp}"
    script_name = f"run_benchx_{suffix}_dynamo_kvrouter.sh"

    # ---- ctx port block ----
    ctx_ports = "\n".join(f"DYN_SYS_PORT_CTX_{i}={8081+i}" for i in range(p))
    # ---- gen port block ----
    gen_ports = "\n".join(
        f"DYN_SYS_PORT_GEN_{i}={port}" for (i, _node, _gpus, port) in decodes
    )

    # ---- decode workers spawn loop (unrolled because each instance differs in node/gpus) ----
    gen_spawn_blocks = []
    for i, node_var, gpus, port in decodes:
        block = f"""# --- decode worker {i} on {node_var} GPUs {gpus} (TP{dtp}) ---
GEN_PORT={port}
echo "[$(date +%H:%M:%S)] Starting gen worker {i} on {node_var} GPUs {gpus} (DYN_SYSTEM_PORT=${{GEN_PORT}})..."
start_bg srun --overlap --ntasks={dtp} --ntasks-per-node={dtp} --nodes=1 --nodelist={node_var} --mpi=pmix \\
  --output="$RESULTS_DIR/gen_worker_g{i}.log" \\
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \\
  --no-container-entrypoint \\
  --no-container-mount-home \\
  bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES={gpus} && $COMMON_ENV && $DYNAMO_WORKER_ENV && \\
    export DYN_SYSTEM_PORT=${{GEN_PORT}} && \\
    trtllm-llmapi-launch python3 -m dynamo.trtllm \\
      --model-path $MODEL_PATH --served-model-name $MODEL \\
      --disaggregation-mode decode \\
      --extra-engine-args $RESULTS_DIR/gen.yaml \\
      --request-plane ${{DYNAMO_REQUEST_PLANE}} \\
      ${{WORKER_METRICS_FLAG}}"
GEN_PIDS+=("${{SRUN_PIDS[-1]}}")"""
        gen_spawn_blocks.append(block)
    gen_spawn = "GEN_PIDS=()\n" + "\n\n".join(gen_spawn_blocks)

    # ---- ctx workers spawn loop (uniform: TP1, NODE0 GPU i) ----
    ctx_gpus = " ".join(str(i) for i in range(p))
    ctx_spawn = f"""CTX_PIDS=()
for GPU in {ctx_gpus}; do
  PORT_VAR="DYN_SYS_PORT_CTX_${{GPU}}"
  CTX_PORT="${{!PORT_VAR}}"
  echo "[$(date +%H:%M:%S)] Starting ctx worker on $NODE0 GPU $GPU (DYN_SYSTEM_PORT=${{CTX_PORT}})..."
  start_bg srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \\
    --output="$RESULTS_DIR/ctx_worker_g${{GPU}}.log" \\
    --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \\
    --no-container-entrypoint \\
    --no-container-mount-home \\
    bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=${{GPU}} && $COMMON_ENV && $DYNAMO_WORKER_ENV && \\
      export DYN_SYSTEM_PORT=${{CTX_PORT}} && \\
      trtllm-llmapi-launch python3 -m dynamo.trtllm \\
        --model-path $MODEL_PATH --served-model-name $MODEL \\
        --disaggregation-mode prefill \\
        --extra-engine-args $RESULTS_DIR/ctx.yaml \\
        --request-plane ${{DYNAMO_REQUEST_PLANE}} \\
        ${{WORKER_METRICS_FLAG}}"
  CTX_PIDS+=("${{SRUN_PIDS[-1]}}")
done"""

    # ---- metrics endpoints + labels ----
    metric_eps = [f"${{NODE0}}:{8081+i}" for i in range(p)]
    metric_labels = [f"ctx_g{i}" for i in range(p)]
    for i, node_var, _gpus, port in decodes:
        # node_var like "$NODE1" -> "${NODE1}"
        nv = "${" + node_var.lstrip("$") + "}"
        metric_eps.append(f"{nv}:{port}")
        metric_labels.append(f"gen_g{i}")
    metric_eps_str = ",".join(metric_eps)
    metric_labels_str = ",".join(metric_labels)

    # ---- NODE2 declaration ----
    node_decl = (
        "NODE0=$(scontrol show hostnames $SLURM_NODELIST | sed -n '1p')\n"
        "NODE1=$(scontrol show hostnames $SLURM_NODELIST | sed -n '2p')"
    )
    if needs_node2:
        node_decl += "\nNODE2=$(scontrol show hostnames $SLURM_NODELIST | sed -n '3p')"

    # ---- layout comment ----
    decode_layout_lines = []
    for i, node_var, gpus, _port in decodes:
        decode_layout_lines.append(
            f"#   {node_var.replace('$','')} — gen worker {i} GPUs {gpus} (TP{dtp})"
        )
    layout_comment = "\n".join(
        [
            "# Layout:",
            f"#   NODE0 — etcd + nats + dynamo frontend + {p} ctx worker(s) (GPUs 0-{p-1}) TP1",
            *decode_layout_lines,
        ]
    )

    # ---- nodes header line ----
    sbatch_nodes = f"#SBATCH --nodes={nnodes}"

    # ---- build via replacements on the 1ctx4gen template ----
    tpl = TEMPLATE_PATH.read_text()

    # SBATCH header
    tpl = tpl.replace(
        "#SBATCH --job-name=core_dlfw_ci-benchx.1ctx4gen.dynamo.kvrouter",
        f"#SBATCH --job-name=core_dlfw_ci-benchx.{suffix}.dynamo.kvrouter",
    )
    tpl = tpl.replace("#SBATCH --nodes=2", sbatch_nodes)
    tpl = tpl.replace(
        "#SBATCH --output=bench/logs/run_benchx_1ctx4gen_dynamo_kvrouter_%j.log",
        f"#SBATCH --output=bench/logs/run_benchx_{suffix}_dynamo_kvrouter_%j.log",
    )
    tpl = tpl.replace(
        "#SBATCH --error=bench/logs/run_benchx_1ctx4gen_dynamo_kvrouter_%j.err",
        f"#SBATCH --error=bench/logs/run_benchx_{suffix}_dynamo_kvrouter_%j.err",
    )

    # Top tagline
    tpl = tpl.replace(
        "# benchx (feat/bench_x sha 11e16c) — 1 ctx + 4 gen with kv router,",
        f"# benchx (feat/bench_x sha 11e16c) — {p} ctx TP1 + {dc} gen TP{dtp} with kv router,",
    )

    # Extend default concurrency sweep up to 160
    tpl = tpl.replace(
        'CONCURRENCY="${CONCURRENCY:-1,2,3,6,8,10,16,32,48,64,80,96,112,128}"',
        'CONCURRENCY="${CONCURRENCY:-1,2,3,6,8,10,16,32,48,64,80,96,112,128,144,160}"',
    )
    tpl = tpl.replace(
        "#                    (default: 1,2,3,6,8,10,16,32,48,64,80,96,112,128)",
        "#                    (default: 1,2,3,6,8,10,16,32,48,64,80,96,112,128,144,160)",
    )

    # Top comment block: replace the 5-line layout
    old_layout = """# Layout:
#   NODE0 — etcd + nats + dynamo frontend + 1 ctx worker(s) (GPUs 0-0)
#   NODE1 — 4 gen worker(s) (GPUs 0-3)"""
    tpl = tpl.replace(old_layout, layout_comment)

    # Submit examples
    tpl = tpl.replace(
        "bench/run_benchx_1ctx4gen_dynamo_kvrouter.sh",
        f"bench/{script_name}",
    )

    # EXP_NAME
    tpl = tpl.replace(
        'EXP_NAME="run_benchx_1ctx4gen_dynamo_kvrouter_${HCTAG}_c${C_TAG}"',
        f'EXP_NAME="run_benchx_{suffix}_dynamo_kvrouter_${{HCTAG}}_c${{C_TAG}}"',
    )

    # NODE declarations
    old_node_decl = (
        "NODE0=$(scontrol show hostnames $SLURM_NODELIST | sed -n '1p')\n"
        "NODE1=$(scontrol show hostnames $SLURM_NODELIST | sed -n '2p')"
    )
    tpl = tpl.replace(old_node_decl, node_decl)

    # Ports block: replace the entire DYN_SYS_PORT_* assignments
    old_ports_block = """DYN_SYS_PORT_CTX_0=8081
DYN_SYS_PORT_GEN_0=8085
DYN_SYS_PORT_GEN_1=8086
DYN_SYS_PORT_GEN_2=8087
DYN_SYS_PORT_GEN_3=8088"""
    new_ports_block = ctx_ports + "\n" + gen_ports
    tpl = tpl.replace(old_ports_block, new_ports_block)

    # gen config source path
    tpl = tpl.replace(
        'GEN_CONFIG_SRC="${REPO_DIR}/bench/gen_config.yaml"',
        f'GEN_CONFIG_SRC="${{REPO_DIR}}/bench/{gen_cfg}"',
    )

    # Gen-worker spawn loop replacement
    old_gen_loop = """# --- 4 gen worker(s) on $NODE1 GPUs 0-3 (decode) ---
GEN_PIDS=()
for GPU in 0 1 2 3; do
  PORT_VAR="DYN_SYS_PORT_GEN_${GPU}"
  GEN_PORT="${!PORT_VAR}"
  echo "[$(date +%H:%M:%S)] Starting gen worker on $NODE1 GPU $GPU (DYN_SYSTEM_PORT=${GEN_PORT})..."
  start_bg srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE1 --mpi=pmix \\
    --output="$RESULTS_DIR/gen_worker_g${GPU}.log" \\
    --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \\
    --no-container-entrypoint \\
    --no-container-mount-home \\
    bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=${GPU} && $COMMON_ENV && $DYNAMO_WORKER_ENV && \\
      export DYN_SYSTEM_PORT=${GEN_PORT} && \\
      trtllm-llmapi-launch python3 -m dynamo.trtllm \\
        --model-path $MODEL_PATH --served-model-name $MODEL \\
        --disaggregation-mode decode \\
        --extra-engine-args $RESULTS_DIR/gen.yaml \\
        --request-plane ${DYNAMO_REQUEST_PLANE} \\
        ${WORKER_METRICS_FLAG}"
  GEN_PIDS+=("${SRUN_PIDS[-1]}")
done"""
    if old_gen_loop not in tpl:
        raise RuntimeError("gen-loop anchor not found")
    tpl = tpl.replace(old_gen_loop, gen_spawn)

    # Ctx-worker spawn loop replacement
    old_ctx_loop = """# --- 1 ctx worker(s) on $NODE0 GPUs 0-0 (prefill) ---
CTX_PIDS=()
for GPU in 0; do
  PORT_VAR="DYN_SYS_PORT_CTX_${GPU}"
  CTX_PORT="${!PORT_VAR}"
  echo "[$(date +%H:%M:%S)] Starting ctx worker on $NODE0 GPU $GPU (DYN_SYSTEM_PORT=${CTX_PORT})..."
  start_bg srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \\
    --output="$RESULTS_DIR/ctx_worker_g${GPU}.log" \\
    --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \\
    --no-container-entrypoint \\
    --no-container-mount-home \\
    bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=${GPU} && $COMMON_ENV && $DYNAMO_WORKER_ENV && \\
      export DYN_SYSTEM_PORT=${CTX_PORT} && \\
      trtllm-llmapi-launch python3 -m dynamo.trtllm \\
        --model-path $MODEL_PATH --served-model-name $MODEL \\
        --disaggregation-mode prefill \\
        --extra-engine-args $RESULTS_DIR/ctx.yaml \\
        --request-plane ${DYNAMO_REQUEST_PLANE} \\
        ${WORKER_METRICS_FLAG}"
  CTX_PIDS+=("${SRUN_PIDS[-1]}")
done"""
    if old_ctx_loop not in tpl:
        raise RuntimeError("ctx-loop anchor not found")
    ctx_block_header = (
        f"# --- {p} ctx worker(s) on $NODE0 GPUs 0-{p-1} (prefill, TP1) ---\n"
    )
    tpl = tpl.replace(old_ctx_loop, ctx_block_header + ctx_spawn)

    # wait_for_dynamo_workers expected counts
    tpl = tpl.replace(
        'echo "[$(date +%H:%M:%S)] Waiting for servers (1 prefill + 4 decode)..."',
        f'echo "[$(date +%H:%M:%S)] Waiting for servers ({p} prefill + {dc} decode)..."',
    )
    tpl = tpl.replace(
        'if ! wait_for_dynamo_workers "${NODE0}" "${FRONTEND_PORT}" 1 4 2700 60; then',
        f'if ! wait_for_dynamo_workers "${{NODE0}}" "${{FRONTEND_PORT}}" {p} {dc} 2700 60; then',
    )

    # Metrics sidecar endpoints/labels
    tpl = tpl.replace(
        '--endpoints "${NODE0}:8081,${NODE1}:8085,${NODE1}:8086,${NODE1}:8087,${NODE1}:8088"',
        f'--endpoints "{metric_eps_str}"',
    )
    tpl = tpl.replace(
        '--labels "ctx_g0,gen_g0,gen_g1,gen_g2,gen_g3"',
        f'--labels "{metric_labels_str}"',
    )

    return tpl, script_name


def main():
    combos = []
    for p in (1, 2, 3, 4):
        for dc in (1, 2, 3):
            combos.append((p, dc, 2))
        combos.append((p, 1, 4))

    for p, dc, dtp in combos:
        content, name = render(p, dc, dtp)
        path = OUT_DIR / name
        path.write_text(content)
        path.chmod(0o755)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
