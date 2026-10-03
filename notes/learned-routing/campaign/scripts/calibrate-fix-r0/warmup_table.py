"""Lifetime-derived open-loop session warm-up per transform (fixer r0, audit F1) -> ../out/warmup_table.json"""
import copy, json, sys
import calib
from sweep import session_template

out = {}
seen = {}
for split in calib.SPLITS:
    for c in calib.candidates(split):
        if c["family"] != "synthetic_sessions" or c["load"]["mode"] != "open_speedup":
            continue
        tag = c["transform_tag"]
        if tag in seen:
            continue
        t = session_template("sessions:s0", "open_speedup")
        t["transform"] = dict(t["transform"], **{k: c["transform"][k] for k in
                                                 ("isl_unique_mult", "isl_prefix_mult", "osl_mult", "prefix_root_mult", "think_mult")})
        w = calib.session_warmup_s(t)
        seen[tag] = calib._WARMUP_MEMO[calib.warmup_key(t)]
        print(tag, json.dumps(seen[tag]))
(calib.FIX / "out" / "warmup_table.json").write_text(json.dumps(
    {"rule": "W = max(180 s, ceil60(q99 of m*sum(think) + 3*sum(E0))) over train seeds 0-3, sessions starting in [0, 2000) s, transform applied",
     "by_transform_tag": seen}, indent=1, sort_keys=True))
