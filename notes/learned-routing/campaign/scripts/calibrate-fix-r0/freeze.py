"""Freeze (stage 2 step 3): install calibrated cells into CR/cells and record facts/test_freeze.json.

usage: freeze.py FINAL_DIR
- copies the current SPLIT_MANIFEST.json to cells/superseded/pre-calibration/ (no deletion);
- writes cells/{train,val,test}.jsonl from FINAL_DIR;
- updates SPLIT_MANIFEST.json with a "calibration" section (cells version lr-cells-v4);
- writes facts/test_freeze.json: SHA-256 of test.jsonl and of every trace (and metadata) it references.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import calib
from learned_routing import HARNESS_VERSION
from learned_routing.cache import bindings_build_id
from learned_routing.cells import Cell

CR = calib.CR
WT = Path("<worktree>")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    final = Path(sys.argv[1])
    cells_dir = CR / "cells"
    # fixer r0 (audit calibration F1): keep calibration r0's frozen artifacts (no deletion)
    sup = cells_dir / "superseded" / "calibration-r0"
    sup.mkdir(parents=True, exist_ok=True)
    for name in ("SPLIT_MANIFEST.json", "train.jsonl", "val.jsonl", "test.jsonl"):
        if not (sup / name).exists():
            shutil.copy2(cells_dir / name, sup / name)
    if not (sup / "test_freeze.json").exists():
        shutil.copy2(CR / "facts" / "test_freeze.json", sup / "test_freeze.json")
    previous_freeze = json.loads((sup / "test_freeze.json").read_text())
    files = {}
    counts = {}
    for split in calib.SPLITS:
        src = final / f"{split}.jsonl"
        dst = cells_dir / f"{split}.jsonl"
        shutil.copy2(src, dst)
        files[split] = {"path": f"cells/{split}.jsonl", "sha256": sha_file(dst)}
        counts[split] = sum(1 for line in dst.read_text().splitlines() if line.strip())
    manifest = json.loads((sup / "SPLIT_MANIFEST.json").read_text())
    previous_calibration = manifest.get("calibration")
    head = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest["calibration"] = {
        "cells_version": "lr-cells-v5",
        "stage": "calibration fixer r0 (audit calibration/independent-rerun r0 F1: lifetime-derived open-loop session warm-up; sessions family re-calibrated; non-session cells byte-identical to lr-cells-v4)",
        "supersedes": {"cells_version": previous_calibration.get("cells_version"),
                       "files": previous_calibration.get("files"),
                       "kept_at": "cells/superseded/calibration-r0/"},
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "files": files,
        "counts": counts,
        "candidates_unchanged": {s: f"cells/{s}.candidates.jsonl" for s in calib.SPLITS},
        "added_test_cells": [c[0] for c in __import__("build_final").ADDED_TEST_CELLS],
        "fix": "audits/calibration/fix-r0.md",
        "rules": "facts/calibration.json",
        "traces_manifest_sha256": sha_file(CR / "traces" / "MANIFEST.json"),
        "worktree_head": head,
        "harness_version": HARNESS_VERSION,
        "previous_split_manifest": "cells/superseded/calibration-r0/SPLIT_MANIFEST.json",
        "pre_calibration_split_manifest": "cells/superseded/pre-calibration/SPLIT_MANIFEST.json",
    }
    manifest["status"] = "calibrated; test.jsonl frozen (facts/test_freeze.json)"
    (cells_dir / "SPLIT_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    # test freeze
    test_path = cells_dir / "test.jsonl"
    traces = {}
    cells = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    for c in cells:
        for rel in c["trace_files"]:
            p = calib.resolve(rel)
            if rel not in traces:
                traces[rel] = {"sha256": sha_file(p), "bytes": p.stat().st_size, "cells": []}
                meta = c.get("derived_meta")
                if meta:
                    mp = calib.resolve(meta)
                    traces[rel]["meta"] = meta
                    traces[rel]["meta_sha256"] = sha_file(mp)
            traces[rel]["cells"].append(c["cell_id"])
            declared = c.get("trace_sha256") or []
            if declared and declared != [traces[rel]["sha256"]]:
                raise SystemExit(f"{c['cell_id']}: trace sha mismatch")
    sources = {}
    for c in cells:
        if c["family"] == "synthetic_sessions":
            src = calib.long_session_source(calib.session_seed(c))
            sources[str(src.relative_to(CR))] = sha_file(src)
        if c["family"] == "agentx":
            meta = json.loads(calib.resolve(c["derived_meta"]).read_text())
            bm = Path(meta["base_manifest"])
            sources[str(bm.relative_to(CR))] = sha_file(bm)
    content_shas = {c["cell_id"]: Cell(raw=c, layout=calib.LAYOUT).content_sha() for c in cells}
    build = bindings_build_id(calib.LAYOUT.cache_dir)
    freeze = {
        "schema": "learned-routing.test-freeze.v1",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "stage": "calibration fixer r0 (re-freeze; no test cell had been evaluated)",
        "supersedes": {"test_jsonl_sha256": previous_freeze["test_jsonl"]["sha256"],
                       "written": previous_freeze.get("written"),
                       "kept_at": "cells/superseded/calibration-r0/test_freeze.json",
                       "reason": "audit calibration/independent-rerun r0 F1: open-loop session warm-up shorter than the session-population relaxation; sessions family re-calibrated"},
        "rule": "Nobody evaluates test cells until the test stage. The test stage verifies every SHA-256 below before its single pass.",
        "test_jsonl": {"path": "cells/test.jsonl", "sha256": files["test"]["sha256"], "cells": counts["test"]},
        "cell_content_sha256": content_shas,
        "traces": traces,
        "trace_sources": sources,
        "engine_json_sha256": sha_file(CR / "config" / "engine.json"),
        "traces_manifest_sha256": sha_file(CR / "traces" / "MANIFEST.json"),
        "split_manifest_sha256": sha_file(cells_dir / "SPLIT_MANIFEST.json"),
        "harness_version": HARNESS_VERSION,
        "bindings_build_id": build["build_id"],
        "worktree_head": head,
        "verify": "python3: hashlib.sha256(open(path,'rb').read()).hexdigest() for cells/test.jsonl and each traces[*] path (relative to CR, 'CR/' prefix stripped)",
    }
    (CR / "facts" / "test_freeze.json").write_text(json.dumps(freeze, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"files": files, "counts": counts, "test_traces": len(traces), "sources": len(sources)}, indent=1))


if __name__ == "__main__":
    main()
