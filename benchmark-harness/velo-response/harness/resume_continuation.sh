#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
JOB=$(cat "$ROOT/control/continuation-job-id")
test "$(cat "$ROOT/control/job-id")" = "$JOB"
read -r PREVIOUS NEXT_LABEL < "$ROOT/control/NEEDS_ALLOCATION"
python3 - "$ROOT" "$JOB" "$NEXT_LABEL" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);job=sys.argv[2];next_label=sys.argv[3]
assert (root/'manifests/nodes.txt').read_text()==(root/'manifests/campaign-original-nodes.txt').read_text(), 'continuation node order changed'
original=json.loads((root/'configs/order-remaining.json').read_text())
remaining=[label for label in original if not (root/'control'/(label+'-COMPLETE')).exists()]
assert remaining and remaining[0]==next_label, (remaining,next_label)
assert all(label.startswith('r') for label in remaining), remaining
(root/'configs'/('order-resume-'+job+'.json')).write_text(json.dumps(remaining,indent=2)+'\n')
print('Remaining runs:',remaining)
PY
srun --jobid="$JOB" --overlap --input=none --nodes=5 --ntasks=5 --ntasks-per-node=1 --cpus-per-task=1 --cpu-bind=none \
    python3 "$ROOT/harness/verify_hardware.py"
python3 - "$ROOT" "$PREVIOUS" "$JOB" <<'PY'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]);previous,job=sys.argv[2:]
old_paths=list((root/'manifests').glob('hardware-'+previous+'-*.json'))
assert len(old_paths)==5, old_paths
for old_path in old_paths:
    old=json.loads(old_path.read_text())
    new=json.loads((root/'manifests'/('hardware-'+job+'-'+old['hostname']+'.json')).read_text())
    assert new['ports']==old['ports'], (old,new)
    assert new['ethernet_mtu']==old['ethernet_mtu']
    assert re.search(r'Speed:.*',new['ethernet'])[0]==re.search(r'Speed:.*',old['ethernet'])[0]
PY
python3 "$ROOT/harness/freeze.py" --root "$ROOT" --verify
mv "$ROOT/control/NEEDS_ALLOCATION" "$ROOT/control/NEEDS_ALLOCATION.$PREVIOUS"
bash "$ROOT/harness/controller.sh" "$ROOT/configs/order-resume-$JOB.json"
python3 "$ROOT/harness/summarize_environment.py"
python3 "$ROOT/harness/report.py"
date -Is > "$ROOT/control/REPORT_COMPLETE"
