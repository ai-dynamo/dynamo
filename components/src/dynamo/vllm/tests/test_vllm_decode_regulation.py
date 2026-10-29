# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-step decode regulation: policy, identity plumbing, and the
skip mixin driven by a fake scheduler with vLLM's running-loop rules."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from dynamo.vllm.decode_regulation import (
    DECODE_REGULATION_METRIC_PREFIX,
    EXTRA_ARGS_KEY,
    Cand,
    DecodeRegulationConfig,
    DecodeRegulationMixin,
    DecodeRegulationPolicy,
    attach_identity,
    cand_from_request,
    identity_from_request,
    vllm_skip_unsupported,
)
from dynamo.vllm.decode_regulation_fit import fit, load

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class FakeRequest:
    def __init__(self, rid, ctx, ident=None, prompt=None, decoded=1):
        self.request_id = rid
        self.num_prompt_tokens = ctx if prompt is None else prompt
        self.num_computed_tokens = ctx
        self.num_output_tokens = decoded
        self.is_prefill_chunk = self.num_computed_tokens < self.num_prompt_tokens
        self.next_decode_eligible_step = 0
        extra = {EXTRA_ARGS_KEY: ident} if ident is not None else None
        self.sampling_params = SimpleNamespace(extra_args=extra)


class FakeV031Scheduler:
    """The parts of vLLM v0.31.0 Scheduler.schedule() the skip relies on."""

    def __init__(self, running):
        self.running = list(running)
        self.current_step = 0
        self.preempted = []

    def schedule(self):
        self.current_step += 1  # first statement of the real schedule()
        tokens = {}
        for req in self.running:
            if self.current_step < req.next_decode_eligible_step:
                continue  # the real loop's decode-skip `continue`
            if req.is_prefill_chunk:
                tokens[req.request_id] = req.num_prompt_tokens - req.num_computed_tokens
            else:
                tokens[req.request_id] = 1
        return SimpleNamespace(
            num_scheduled_tokens=tokens, total_num_scheduled_tokens=sum(tokens.values())
        )

    def update_from_output(self, out, _runner_output):
        for req in self.running:
            k = out.num_scheduled_tokens.get(req.request_id)
            if k:
                req.num_computed_tokens += k
                req.is_prefill_chunk = req.num_computed_tokens < req.num_prompt_tokens
                if not req.is_prefill_chunk:
                    req.num_output_tokens += 1

    def shutdown(self):
        pass


class Sched(DecodeRegulationMixin, FakeV031Scheduler):
    def __init__(self, running, cfg):
        FakeV031Scheduler.__init__(self, running)
        self._reg_init(cfg)


def cfg(mode="class_blind", **kw):
    c = DecodeRegulationConfig(mode=mode, log_s=1e9)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


AGENT = {"interactive": False}


def overloaded(n_subs=8, ctx=100_000):
    """One interactive chat, an agent's main session, and n_subs subagents."""
    reqs = [FakeRequest("chat", 20_000), FakeRequest("main", ctx, AGENT)]
    reqs += [FakeRequest(f"s{i}", ctx, AGENT) for i in range(n_subs)]
    return reqs


def step(s):
    out = s.schedule()
    s.update_from_output(out, None)
    return set(out.num_scheduled_tokens)


# --- the skip mechanism ---------------------------------------------------


def test_off_mode_schedules_everything():
    s = Sched(overloaded(), cfg("off"))
    assert step(s) == {r.request_id for r in s.running}


def test_no_skips_when_batch_fits():
    s = Sched(overloaded(n_subs=2, ctx=10_000), cfg())
    assert step(s) == {r.request_id for r in s.running}


def test_overload_skips_agents_but_keeps_them_running():
    reqs = overloaded()
    s = Sched(reqs, cfg())
    scheduled = step(s)
    assert {"chat", "main"} <= scheduled
    skipped = {r.request_id for r in reqs} - scheduled
    # base 6.95 ms, limit 16.17 ms, each 100k-token agent adds ~1.83 ms: 5 of 9 fit
    assert len(skipped) == 4
    assert s.running == reqs  # never removed from running: KV stays resident


def test_skip_lasts_exactly_one_step():
    reqs = overloaded()
    s = Sched(reqs, cfg())
    s._reg_apply()
    skipped = [r for r in reqs if r.next_decode_eligible_step > 0]
    assert skipped
    assert all(r.next_decode_eligible_step == s.current_step + 2 for r in skipped)
    s._reg_policy.decide([])  # forget the decision; the field alone must expire
    out = FakeV031Scheduler.schedule(s)  # the step the skip was for
    assert not {r.request_id for r in skipped} & set(out.num_scheduled_tokens)
    out = FakeV031Scheduler.schedule(s)  # the next step: eligible again
    assert {r.request_id for r in skipped} <= set(out.num_scheduled_tokens)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_progress_bound_every_k_steps(k):
    reqs = overloaded(n_subs=16)
    s = Sched(reqs, cfg(k=k))
    s._reg_policy.observe = lambda *_: None  # fake steps take microseconds
    last = {r.request_id: 0 for r in reqs}
    skipped = 0
    for i in range(1, 60):
        ran = step(s)
        skipped += len(reqs) - len(ran)
        for rid in ran:
            assert i - last[rid] <= k, (rid, i, last[rid])
            last[rid] = i
    assert all(60 - v <= k for v in last.values())
    assert (skipped > 0) == (k > 1)  # K=1 forces every decode


def test_prefill_requests_are_never_skipped():
    reqs = overloaded() + [FakeRequest("p", 100_000, AGENT, prompt=200_000)]
    s = Sched(reqs, cfg())
    assert "p" in step(s)


def test_requests_without_identity_are_never_skipped():
    reqs = [FakeRequest(f"x{i}", 100_000) for i in range(12)]
    s = Sched(reqs, cfg())
    assert step(s) == {r.request_id for r in reqs}


# --- policy ----------------------------------------------------------------


def cands(n_subs=8, ctx=100_000):
    cs = [Cand("chat", 20_000), Cand("main", ctx, interactive=False)]
    return cs + [Cand(f"s{i}", ctx, interactive=False) for i in range(n_subs)]


def test_any_regulated_request_may_be_skipped():
    p = DecodeRegulationPolicy(cfg("class_blind"))
    skipped = set()
    for _ in range(4):
        skipped |= p.decide(cands(n_subs=12)).skip
    assert "main" in skipped
    assert "chat" not in skipped


def test_residual_target_sets_the_step_budget():
    p = DecodeRegulationPolicy(cfg("class_blind"))
    fixed = p.decide(cands(n_subs=12))
    loose = cands(n_subs=12)
    tight = cands(n_subs=12)
    for c in loose:
        if c.interactive:
            c.tau_ms = 2 * p.cfg.tau_interactive_ms
    for c in tight:
        if c.interactive:
            c.tau_ms = 0.0  # already late: nothing opportunistic
    d_loose = DecodeRegulationPolicy(cfg("class_blind")).decide(loose)
    d_tight = DecodeRegulationPolicy(cfg("class_blind")).decide(tight)
    assert d_loose.tau_ms == 2 * p.cfg.tau_interactive_ms
    assert len(d_loose.skip) < len(fixed.skip) < len(d_tight.skip)
    assert d_tight.n_opp == 0


def test_deadline_target_from_arrival_and_max_tokens():
    c = DecodeRegulationConfig(
        mode="class_blind", deadline=(500.0, 18.48), slack_margin=1.0
    )
    r = FakeRequest("chat", 20_000, decoded=4)
    r.arrival_time = 100.0
    r.sampling_params.max_tokens = 64
    # 200 ms in: (500 + 18.48 * 64 - 200) ms left for 60 tokens
    got = cand_from_request(r, c, now=100.2).tau_ms
    assert abs(got - (500 + 18.48 * 64 - 200) / 60) < 1e-6
    agent = FakeRequest("a", 20_000, ident=AGENT)
    agent.arrival_time, agent.sampling_params.max_tokens = 100.0, 64
    assert cand_from_request(agent, c, now=100.2).tau_ms is None
    plain = DecodeRegulationConfig(mode="class_blind")
    assert cand_from_request(r, plain, now=100.2).tau_ms is None


def test_config_deadline_from_env():
    c = DecodeRegulationConfig.from_env(
        {
            "DYN_DECODE_REGULATION_MODE": "class_blind",
            "DYN_DECODE_REGULATION_DEADLINE": "500,18.48",
            "DYN_DECODE_REGULATION_SLACK_MARGIN": "0.8",
        }
    )
    assert c.deadline == (500.0, 18.48) and c.slack_margin == 0.8
    assert DecodeRegulationConfig.from_env({}).deadline is None


def test_coarse_guard_caps_opportunistic_context():
    p = DecodeRegulationPolicy(cfg(max_opp_ctx=150_000))
    d = p.decide(cands())
    assert d.n_opp == 1


def test_bias_tracks_slower_steps_and_is_clamped():
    p = DecodeRegulationPolicy(cfg())
    for _ in range(200):
        p.observe(10.0 * p.bias, 30.0)
    assert p.bias == 2.0
    before = p.decide(cands()).n_opp
    p.bias = 1.0
    assert p.decide(cands()).n_opp >= before


# --- identity plumbing -------------------------------------------------------


@pytest.mark.parametrize(
    ("ctx", "expected"),
    [
        (None, {"interactive": True}),
        ({"session_id": "interactive-3"}, {"interactive": True}),
        ({"session_id": "root"}, {"interactive": False}),
        ({"session_id": "child", "parent_session_id": "root"}, {"interactive": False}),
    ],
)
def test_identity_from_request(ctx, expected):
    req = {} if ctx is None else {"agent_context": ctx}
    assert identity_from_request(req, "interactive-") == expected


@pytest.mark.parametrize(
    ("req", "expected"),
    [
        ({"routing": {"priority": 5}}, {"interactive": True}),
        ({"routing": {"priority": 1}}, {"interactive": True}),
        ({}, {"interactive": False}),  # unset priority is 0
        ({"routing": {"priority": "x"}}, {"interactive": False}),
        (
            {"routing": {"priority": 5}, "agent_context": {"session_id": "root"}},
            {"interactive": True},
        ),
        (
            {"agent_context": {"session_id": "c", "parent_session_id": "root"}},
            {"interactive": False},
        ),
        # the session prefix rule does not apply in priority mode
        ({"agent_context": {"session_id": "interactive-3"}}, {"interactive": False}),
    ],
)
def test_identity_from_priority(req, expected):
    assert identity_from_request(req, "interactive-", priority_min=1) == expected


def test_summary_counts_both_classes():
    reqs = overloaded(n_subs=2, ctx=10_000)
    s = Sched(reqs, cfg("class_blind", priority_min=1))
    for r in reqs[1:]:
        r.sampling_params.extra_args = {EXTRA_ARGS_KEY: {"interactive": False}}
    step(s)
    line = s._reg_stats.summary_and_reset(1.0)
    assert "interactive=1 regulated=3 " in line


def test_prometheus_series():
    prometheus_client = pytest.importorskip("prometheus_client")
    registry = prometheus_client.CollectorRegistry()
    reqs = overloaded()
    s = Sched([], cfg(failsafe_steps=2))
    s._reg_init(
        cfg(failsafe_steps=2), model_name="m", dp_rank=1, metrics_registry=registry
    )
    s.running = reqs
    step(s)

    def value(name):
        return registry.get_sample_value(
            f"{DECODE_REGULATION_METRIC_PREFIX}_{name}", {"model": "m", "dp_rank": "1"}
        )

    assert value("steps_total") == 1
    assert value("engaged_steps_total") == 1
    assert value("interactive_total") == 1
    assert value("regulated_total") == len(reqs) - 1
    assert value("skipped_total") == 4  # as in the overload test above
    assert value("enabled") == 1
    s._reg_policy.bias = 2.0
    s._reg_failsafe()
    s._reg_failsafe()
    assert value("enabled") == 0


def test_failsafe_disables_when_bias_stays_pinned():
    s = Sched(overloaded(), cfg(failsafe_steps=3))
    s._reg_policy.bias = 2.0
    for _ in range(2):
        s._reg_failsafe()
    assert s._reg_cfg.enabled
    s._reg_failsafe()
    assert not s._reg_cfg.enabled
    assert step(s) == {r.request_id for r in s.running}  # stock scheduling


def test_unwritable_trace_path_does_not_stop_the_scheduler(tmp_path):
    s = Sched(overloaded(), cfg(trace_path=str(tmp_path / "missing" / "t.jsonl")))
    assert s._reg_trace is None and s._reg_cfg.enabled


def test_failsafe_leaves_observe_mode_tracing():
    s = Sched(overloaded(), cfg("observe", failsafe_steps=3))
    s._reg_policy.bias = 2.0
    for _ in range(5):
        s._reg_failsafe()
    assert s._reg_cfg.mode == "observe"


def test_failsafe_resets_when_bias_recovers():
    s = Sched(overloaded(), cfg(failsafe_steps=3))
    for b in (2.0, 2.0, 1.2, 2.0, 2.0):
        s._reg_policy.bias = b
        s._reg_failsafe()
    assert s._reg_cfg.enabled


def test_config_priority_from_env():
    c = DecodeRegulationConfig.from_env(
        {
            "DYN_DECODE_REGULATION_MODE": "class_blind",
            "DYN_DECODE_REGULATION_PRIORITY_MIN": "1",
        }
    )
    assert c.priority_min == 1
    assert (
        DecodeRegulationConfig.from_env(
            {"DYN_DECODE_REGULATION_MODE": "class_blind"}
        ).priority_min
        is None
    )


def test_attach_identity_keeps_other_extra_args():
    sp = SimpleNamespace(extra_args={"kv_transfer_params": {"x": 1}})
    attach_identity(sp, {"agent_context": {"session_id": "root"}}, cfg())
    assert sp.extra_args["kv_transfer_params"] == {"x": 1}
    assert sp.extra_args[EXTRA_ARGS_KEY] == {"interactive": False}
    sp = SimpleNamespace(extra_args=None)
    attach_identity(sp, {}, cfg("off"))
    assert sp.extra_args is None


def test_config_from_env():
    c = DecodeRegulationConfig.from_env(
        {
            "DYN_DECODE_REGULATION_MODE": "class_blind",
            "DYN_DECODE_REGULATION_K": "3",
            "DYN_DECODE_REGULATION_COEFFS": "1,2,0.5",
        }
    )
    assert (c.mode, c.k, c.coeffs) == ("class_blind", 3, (1.0, 2.0, 0.5))
    assert not DecodeRegulationConfig.from_env({}).enabled
    with pytest.raises(ValueError):
        DecodeRegulationConfig.from_env({"DYN_DECODE_REGULATION_MODE": "fastest"})
    with pytest.raises(ValueError):
        DecodeRegulationConfig.from_env(
            {
                "DYN_DECODE_REGULATION_MODE": "class_blind",
                "DYN_DECODE_REGULATION_K": "0",
            }
        )
    # Off ignores the other settings, so a bad value can't break a worker.
    assert not DecodeRegulationConfig.from_env(
        {"DYN_DECODE_REGULATION_K": "0", "DYN_DECODE_REGULATION_COEFFS": "x"}
    ).enabled
    on = DecodeRegulationConfig.from_env({"DYN_DECODE_REGULATION_MODE": "on"})
    assert on.mode == "class_blind"


def test_step_timing_ignores_empty_outputs():
    s = Sched(overloaded(), cfg())
    out = s.schedule()
    s._reg_on_update(SimpleNamespace())  # an empty output never recorded
    assert len(s._reg_pending) == 1
    s.update_from_output(out, None)
    assert not s._reg_pending


def test_prefill_in_the_step_reduces_opportunistic_decodes():
    without = DecodeRegulationPolicy(cfg()).decide(cands()).n_opp
    p = DecodeRegulationPolicy(cfg(prefill_coeffs=(1.0, 0.0005)))
    assert p.decide(cands(), prefill_tokens=8192).n_opp < without


def test_identity_cache_follows_running_set():
    reqs = overloaded()
    s = Sched(reqs, cfg("class_blind"))
    step(s)
    assert set(s._reg_static) == {r.request_id for r in reqs}
    # identity is read once: a later change to extra_args is not seen
    reqs[1].sampling_params.extra_args = None
    assert s._reg_static["main"].interactive is False
    for k in range(10):  # finished requests are dropped, in batches
        s.running = [FakeRequest(f"n{k}-{i}", 1_000) for i in range(20)]
        step(s)
        assert len(s._reg_static) <= 2 * 20 + 64 + 20
    assert "main" not in s._reg_static


def test_observe_mode_decides_but_never_skips():
    reqs = overloaded()
    s = Sched(reqs, cfg("observe"))
    assert step(s) == {r.request_id for r in reqs}
    line = s._reg_stats.summary_and_reset(1.0)
    assert "engaged=1 skipped=4 " in line  # what class_blind would have skipped


def test_fit_recovers_step_model_from_trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    with open(path, "w") as f:
        for i in range(200):
            n, ctx = 1 + i % 50, 10_000 * (1 + i % 37)
            actual = 5.0 + 0.2 * n + 1.5e-5 * ctx
            f.write(
                f'{{"t": {i * 0.03}, "n": {n}, "ctx": {ctx}, '
                f'"decode_only": {"true" if i % 5 else "false"}, "actual": {actual}}}\n'
            )
    rows = load(str(path), skip_first_s=0.0, max_ms=200.0)
    assert len(rows) == 160  # mixed prefill steps are dropped
    a, b, c = fit(rows)
    assert abs(a - 5.0) < 1e-6 and abs(b - 0.2) < 1e-8 and abs(c - 1.5e-5) < 1e-10


def test_no_regulation_without_an_interactive_request():
    reqs = overloaded(n_subs=16)[1:]  # agents only, far over any target
    s = Sched(reqs, cfg())
    assert step(s) == {r.request_id for r in reqs}
    d = DecodeRegulationPolicy(cfg()).decide(cands(n_subs=16)[1:])
    assert not d.engaged and not d.skip


# --- vLLM internals the skip relies on --------------------------------------


class _Req:
    def __init__(self):
        self.next_decode_eligible_step = 0


class _ReqWithoutField:
    def __init__(self):
        self.num_computed_tokens = 0


class _PluginReq(_Req):
    """A plugin's Request subclass, as vLLM-Omni's OmniRequest."""

    def __init__(self):
        super().__init__()
        self.extra = None


class _PluginReqWithoutSuper(_Req):
    def __init__(self):
        self.extra = None


class _Sched:
    def schedule(self):
        self.current_step += 1
        for request in self.running:
            if self.current_step < request.next_decode_eligible_step:
                continue


class _SchedWithoutCheck:
    def schedule(self):
        self.current_step += 1


class _SchedIncrementsLate:
    def schedule(self):
        for request in self.running:
            if self.current_step < request.next_decode_eligible_step:
                continue
        self.current_step += 1


def test_skip_support_check_accepts_v031_shape():
    assert vllm_skip_unsupported(_Sched, _Req) == ""


def test_skip_support_check_accepts_plugin_request_subclass():
    assert vllm_skip_unsupported(_Sched, _PluginReq) == ""


@pytest.mark.parametrize(
    "sched,req,reason",
    [
        (_Sched, _ReqWithoutField, "no next_decode_eligible_step"),
        (_Sched, _PluginReqWithoutSuper, "does not call super().__init__"),
        (_SchedWithoutCheck, _Req, "no longer checks"),
        (_SchedIncrementsLate, _Req, "after the skip check"),
        (_Sched, dict, "cannot read"),
    ],
)
def test_skip_support_check_disables_on_changed_vllm(sched, req, reason):
    assert reason in vllm_skip_unsupported(sched, req)


def test_requests_without_the_field_disable_regulation():
    reqs = overloaded()
    s = Sched(reqs, cfg())
    for r in reqs:
        del r.next_decode_eligible_step
    s._reg_apply()
    assert not s._reg_cfg.enabled


_INSTALLED_CHECK = """
from vllm.plugins import load_general_plugins
load_general_plugins()  # as the engine does; vLLM-Omni swaps in a Request subclass
import vllm.v1.core.sched.scheduler as s, vllm.v1.request as r
from dynamo.vllm.decode_regulation import vllm_skip_unsupported
print("RESULT=" + vllm_skip_unsupported(s.Scheduler, r.Request))
"""


def test_installed_vllm_supports_the_skip():
    """Fails on a vLLM upgrade, or a vLLM plugin, that changes the internals the
    skip relies on. Runs in a subprocess so loading plugins leaves this one alone."""
    out = subprocess.run(
        [sys.executable, "-c", _INSTALLED_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    result = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT=")]
    assert result == ["RESULT="], out.stdout[-2000:]
