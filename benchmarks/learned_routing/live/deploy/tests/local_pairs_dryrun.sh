#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Workstation dry run of job_node.sh's pairs mode (no Slurm, GPU, vLLM or frontend). The real
# job_node.sh, common.sh and pairs.py run with stubbed srun/scontrol, setup scripts, workers,
# frontend, health checks and payload, so the control flow is exercised end to end:
#   1. three pairs, the third too long for the job's remaining time: pairs 1-2 run in file order,
#      each with its own check and serve frontend and exactly its own input, LR_RUN_TIMEOUT_S
#      = 1.5 x est_s + 300; pair 3 is recorded as skipped_time; the job exits non-zero; the runtime
#      manifest lists all three pairs;
#   2. a pairs file naming a policy outside the plan fails before node prep.
# Usage: tests/local_pairs_dryrun.sh OUT_DIR
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deploy="$(dirname "$here")"
[[ $# -eq 1 ]] || { echo "usage: $0 OUT_DIR" >&2; exit 2; }
out=$1
[[ ! -e "$out" ]] || { echo "error: $out exists" >&2; exit 2; }
mkdir -p "$out"/{bin,deploy,plan/policies/pol_a,plan/policies/pol_b,inputs,logs}
out="$(cd "$out" && pwd)"

# Stubs on PATH: srun runs its command locally (skipping --options; the container-create step is
# a no-op); scontrol reports one node and an EndTime STUB_JOB_SECONDS from now.
cat > "$out/bin/srun" <<'SH'
#!/usr/bin/env bash
args=()
for a in "$@"; do
  case "$a" in
    --container-image=*) exit 0 ;;
    --*) ;;
    *) args+=("$a") ;;
  esac
done
exec "${args[@]}"
SH
cat > "$out/bin/scontrol" <<'SH'
#!/usr/bin/env bash
case "$1 $2" in
  "show hostnames") echo localnode ;;
  "show job") echo "JobId=$3 JobState=RUNNING EndTime=$(date -d "@$STUB_JOB_END" +%Y-%m-%dT%H:%M:%S) Partition=stub" ;;
esac
SH
chmod +x "$out/bin/srun" "$out/bin/scontrol"

cp "$deploy/job_node.sh" "$deploy/common.sh" "$deploy/pairs.py" "$out/deploy/"
for s in node_prep.sh stage_weights.sh build_env.sh; do
  printf '#!/usr/bin/env bash\necho "%s $*" >> "%s/logs/setup.log"\n' "$s" "$out" > "$out/deploy/$s"
done
printf '#!/usr/bin/env bash\nexec sleep 600\n' > "$out/deploy/node_workers.sh"
printf '#!/usr/bin/env bash\necho "frontend $2 $3" >> "%s/logs/frontend.log"\nexec sleep 600\n' "$out" \
  > "$out/deploy/frontend.sh"
cat > "$out/deploy/health_check.py" <<PY
import sys
open("$out/logs/hc.log", "a").write(" ".join(sys.argv[1:2]) + "\n")
PY
cat > "$out/payload.sh" <<SH
#!/usr/bin/env bash
echo "\$LR_POLICY_SLUG \$LR_PAIR_LABEL \$LR_SMOKE_RUNS \$LR_RUN_TIMEOUT_S" >> "$out/logs/payload.log"
SH

printf '{"model": "Qwen/Qwen3-32B", "block_size": 16}\n' > "$out/plan/engine_plan.json"
printf '{"plan_id": "dryrun", "replicate": 0}\n' > "$out/plan/PLAN.json"
for slug in pol_a pol_b; do echo '{}' > "$out/plan/policies/$slug/policy_plan.json"; done
for run in c1__a c1__b c1__a2; do
  mkdir -p "$out/inputs/$run"
  printf '{"cell_id": "c1", "k": 0, "num_workers": 4, "salt": "s-%s"}\n' "$run" > "$out/inputs/$run/manifest.json"
  : > "$out/inputs/$run/aiperf_input.jsonl"
done
printf 'a pol_a c1__a 10\nb pol_b c1__b 10\n# comment\na2 pol_a c1__a2 100000\n' > "$out/pairs.ok"
printf 'a pol_a c1__a 10\nz pol_z c1__b 10\n' > "$out/pairs.bad"

run_job() {
  local tag=$1 pairs=$2
  cat > "$out/$tag.env" <<ENV
LR_COMMIT=$(printf '0%.0s' {1..40})
LR_PLAN_DIR=$out/plan
LR_DEPLOY_DIR=$out/deploy
LR_RUN_TAG=$tag
LR_PAIRS=$pairs
LR_PAYLOAD=$out/payload.sh
LR_SMOKE_INPUTS=$out/inputs
LR_PAIR_MARGIN_S=60
LR_RUNS_ROOT=$out/runs
LR_NODE_ROOT=$out/node
LR_CACHE_ROOT=$out/cache
LR_LIVE_ROOT=$out/live
LR_SHARED_ROOT=$out/shared
LR_IMAGE_SQSH=$out/shared/image.sqsh
LR_MODEL_MASTER=$out/shared/model
LR_TOOLCHAIN_ENV=$out/shared/toolchain-env.sh
LR_FRONTEND_CPUS=0
LR_CLIENT_CPUS=0
LR_AUX_CPUS=0
ENV
  set +e
  PATH="$out/bin:$PATH" SLURM_JOB_ID="$tag" SLURM_JOB_NODELIST=localnode SLURM_CPUS_ON_NODE=1 \
    STUB_JOB_END=$(($(date +%s) + 3600)) LR_JOB_ENV="$out/$tag.env" \
    bash "$out/deploy/job_node.sh" > "$out/logs/$tag.stdout" 2>&1
  local rc=$?
  set -e
  echo "$rc"
}

rc_ok=$(run_job 1001 "$out/pairs.ok")
rc_bad=$(run_job 1002 "$out/pairs.bad")
python3 - "$out" "$rc_ok" "$rc_bad" <<'PY'
import json, sys
from pathlib import Path
out, rc_ok, rc_bad = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
run = out / "runs" / "1001-1001"
status = {p.parent.name: json.loads(p.read_text())["status"] for p in sorted(run.glob("pairs/*/status.json"))}
want = {"01-a": "ok", "02-b": "ok", "03-a2": "skipped_time"}
assert status == want, status
assert rc_ok != 0, "a skipped pair must fail the job"
payload = (out / "logs/payload.log").read_text().splitlines()
assert payload == ["pol_a a c1__a 315", "pol_b b c1__b 315"], payload
fe = [line.split()[1] + ":" + Path(line.split()[2]).name for line in (out / "logs/frontend.log").read_text().splitlines()]
assert fe == ["pol_a:check", "pol_a:serve", "pol_b:check", "pol_b:serve"], fe
manifest = json.loads((run / "runtime-manifest.json").read_text())
assert sorted(manifest["pairs"]) == sorted(want), manifest["pairs"]
assert manifest["failures"] == ["03-a2:skipped_time"], manifest["failures"]
assert rc_bad != 0
assert "/1002-1002" not in (out / "logs/setup.log").read_text(), "bad pairs must fail before node prep"
assert "is not in the plan" in (out / "logs/1002.stdout").read_text()
print(json.dumps({"ok": True, "pairs": status, "rc_ok": rc_ok, "rc_bad": rc_bad}))
PY
