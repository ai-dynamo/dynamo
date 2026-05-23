// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// KVBM Hub web UI — single Alpine.js component.
//
// Same-origin against the hub's control port. Phase B's `GET /modules` and
// Phase C's `GET /describe` (push-authoritative, 503 describe_pending when
// the leader hasn't pushed yet) supply all the data. `POST /control/...`
// drives the action buttons.
//
// Registered via `alpine:init` so it works regardless of script-load
// ordering relative to Alpine's auto-start.

document.addEventListener("alpine:init", () => {
  window.Alpine.data("hubApp", hubAppData);
});

function hubAppData() {
  return {
    // ---------- state ----------
    instances: [],
    selectedId: null,
    selected: null,           // { instance_id, role, host, ... }  (from /describe.description, when present)
    modules: [],              // [ "core", "transfer", ... ]      (cached or freshly fetched)
    describe: null,           // InstanceDescription              (null when pending or not selected)
    describeSource: null,     // "push" | "pull_fallback"
    describeAgeSecs: 0,
    describePending: false,
    describePendingHint: "",
    actionResults: {},        // last action result string per id

    // Metrics tab — on-demand only by default.
    metrics: null,            // MetricsSnapshotResponse                    (or null)
    metricsFetchedAt: 0,      // browser-side ms timestamp of last successful fetch
    metricsError: null,       // last error string, cleared on success
    metricsAutoRefresh: false,
    metricsTimer: null,

    instancesTimer: null,
    detailTimer: null,
    pendingPollMs: 2000,
    cachedPollMs: 30000,
    metricsAutoMs: 5000,

    // ---------- lifecycle ----------
    init() {
      this.refreshInstances();
      this.instancesTimer = setInterval(() => this.refreshInstances(), 5000);
    },

    // ---------- instances list ----------
    async refreshInstances() {
      try {
        const r = await fetch("/v1/instances");
        if (!r.ok) return;
        const body = await r.json();
        // Body: { instances: [PeerInfo, ...] } — preserve order by id.
        // Annotate with role from cached describe (if we have it for selected).
        this.instances = (body.instances || []).map(p => {
          const id = (p.instance_id || p.id || "");
          const role = (this.selected && this.selected.instance_id === id && this.selected.role) || null;
          return { instance_id: id, role };
        });
        // Auto-select first instance on initial load.
        if (this.selectedId === null && this.instances.length > 0) {
          this.select(this.instances[0].instance_id);
        }
        // Drop selection if the instance went away.
        if (this.selectedId && !this.instances.some(i => i.instance_id === this.selectedId)) {
          this.deselect();
        }
      } catch (e) {
        console.warn("refreshInstances failed:", e);
      }
    },

    // ---------- selection ----------
    select(id) {
      if (this.selectedId === id) return;
      this.selectedId = id;
      this.selected = { instance_id: id };
      this.modules = [];
      this.describe = null;
      this.describeSource = null;
      this.describeAgeSecs = 0;
      this.describePending = false;
      this.describePendingHint = "";
      this.actionResults = {};
      // Reset metrics state per-selection so stale numbers never leak across
      // instances. Auto-refresh is sticky across selections (it's a UI mode,
      // not per-instance state).
      this.metrics = null;
      this.metricsFetchedAt = 0;
      this.metricsError = null;
      this.stopMetricsTimer();
      this.refreshDetail();
      if (this.detailTimer) clearInterval(this.detailTimer);
      this.detailTimer = setInterval(() => this.refreshDetail(), this.pendingPollMs);
      if (this.metricsAutoRefresh) this.startMetricsTimer();
    },

    deselect() {
      this.selectedId = null;
      this.selected = null;
      this.metrics = null;
      this.metricsError = null;
      this.stopMetricsTimer();
      if (this.detailTimer) { clearInterval(this.detailTimer); this.detailTimer = null; }
    },

    // ---------- detail refresh ----------
    async refreshDetail() {
      if (!this.selectedId) return;
      await Promise.all([this.refreshModules(), this.refreshDescribe()]);
      // Slow the polling once describe lands.
      const targetMs = this.describePending ? this.pendingPollMs : this.cachedPollMs;
      if (this.detailTimer) {
        clearInterval(this.detailTimer);
        this.detailTimer = setInterval(() => this.refreshDetail(), targetMs);
      }
    },

    async refreshModules() {
      try {
        const r = await fetch(`/v1/instances/${this.selectedId}/modules`);
        if (!r.ok) { this.modules = []; return; }
        const body = await r.json();
        this.modules = body.modules || [];
      } catch (e) {
        console.warn("refreshModules failed:", e);
      }
    },

    async refreshDescribe() {
      try {
        const r = await fetch(`/v1/instances/${this.selectedId}/describe`);
        if (r.status === 503) {
          const body = await r.json().catch(() => ({}));
          this.describePending = true;
          this.describe = null;
          const secs = body.registered_secs_ago;
          this.describePendingHint = (secs != null)
            ? `registered ${secs}s ago — workers may still be stamping their layouts`
            : "leader has not yet pushed describe";
          return;
        }
        if (!r.ok) return;
        const body = await r.json();
        this.describe = body.description;
        this.describeSource = body.source;
        this.describeAgeSecs = body.age_secs || 0;
        this.describePending = false;
        this.describePendingHint = "";
        // Mirror role onto `selected` and the sidebar entry so the chip renders.
        if (this.selected) {
          this.selected.role = this.describe.role || null;
          this.selected.host = this.describe.host || null;
        }
        const entry = this.instances.find(i => i.instance_id === this.selectedId);
        if (entry) entry.role = this.describe.role || null;
      } catch (e) {
        console.warn("refreshDescribe failed:", e);
      }
    },

    async forceDescribe() {
      if (!this.selectedId) return;
      try {
        const r = await fetch(`/v1/instances/${this.selectedId}/describe?force=true`);
        if (r.ok) {
          const body = await r.json();
          this.describe = body.description;
          this.describeSource = body.source;
          this.describeAgeSecs = body.age_secs || 0;
          this.describePending = false;
        }
      } catch (e) {
        console.warn("forceDescribe failed:", e);
      }
    },

    // ---------- topology helpers ----------
    topologyRows() {
      if (!this.describe || !this.describe.workers) return [];
      const rows = [];
      for (const w of this.describe.workers) {
        const rank = w.parallelism ? w.parallelism.rank : "—";
        if (!w.layouts || w.layouts.length === 0) {
          rows.push({
            key: `${w.worker_id}-empty`,
            worker: `${w.worker_id}`,
            rank,
            tier: "—",
            location: "—",
            shape: "(no layouts stamped)",
            layout: "—",
            blocks: "0",
          });
          continue;
        }
        for (const l of w.layouts) {
          const c = l.config;
          const shape = `${c.num_layers}×${c.outer_dim}×${c.page_size}×${c.inner_dim} (${c.dtype_width_bytes}B)`;
          rows.push({
            key: `${w.worker_id}-${l.tier}`,
            worker: `${w.worker_id}`,
            rank,
            tier: l.tier,
            location: this.fmtLocation(l.location),
            shape,
            layout: `${l.layout_type}/${l.block_layout}`,
            blocks: `${c.num_blocks}`,
          });
        }
      }
      return rows;
    },

    fmtLocation(loc) {
      if (loc == null) return "—";
      if (typeof loc === "string") return loc;
      // Tagged-variants: { device: 0 } / { disk: 1234 } / "system" / "pinned"
      const key = Object.keys(loc)[0];
      return `${key}(${loc[key]})`;
    },

    fmtBytes(n) {
      if (n == null) return "—";
      const units = ["B", "KiB", "MiB", "GiB", "TiB"];
      let i = 0; let v = Number(n);
      while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
      return v >= 100 ? `${v.toFixed(0)} ${units[i]}` : `${v.toFixed(1)} ${units[i]}`;
    },

    shortId(id) {
      if (!id) return "—";
      const s = String(id);
      return s.length > 13 ? s.slice(0, 8) + "…" + s.slice(-4) : s;
    },

    // ---------- config tree ----------
    renderConfigTree(v) {
      const esc = (s) => String(s).replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
      }[c]));
      const render = (val) => {
        if (val === null) return `<span class="v nil">null</span>`;
        if (typeof val === "boolean") return `<span class="v bool">${val}</span>`;
        if (typeof val === "number") return `<span class="v num">${val}</span>`;
        if (typeof val === "string") return `<span class="v">${esc(JSON.stringify(val))}</span>`;
        if (Array.isArray(val)) {
          if (val.length === 0) return `<span class="v nil">[]</span>`;
          return `<details open><summary>[${val.length}]</summary><ul>${
            val.map((e, i) => `<li><span class="k">${i}:</span> ${render(e)}</li>`).join("")
          }</ul></details>`;
        }
        if (typeof val === "object") {
          const keys = Object.keys(val);
          if (keys.length === 0) return `<span class="v nil">{}</span>`;
          return `<details open><summary>{${keys.length}}</summary><ul>${
            keys.map(k => `<li><span class="k">${esc(k)}:</span> ${render(val[k])}</li>`).join("")
          }</ul></details>`;
        }
        return esc(String(val));
      };
      if (!v) return `<span class="v nil">(no config)</span>`;
      return render(v);
    },

    // ---------- actions ----------
    async devReset() {
      const r = await fetch(`/v1/instances/${this.selectedId}/control/dev/reset`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      const body = await r.text();
      this.actionResults.dev_reset = `${r.status}: ${body.slice(0, 200)}`;
    },

    // ---------- metrics ----------
    async refreshMetrics() {
      if (!this.selectedId) return;
      try {
        const r = await fetch(`/v1/instances/${this.selectedId}/control/metrics/snapshot`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: "{}",
        });
        if (!r.ok) {
          // The body shape is `{ error, kind }` for ControlError responses.
          const body = await r.json().catch(() => ({}));
          this.metrics = null;
          this.metricsError = `${r.status}: ${body.error || body.kind || "fetch failed"}`;
          return;
        }
        const body = await r.json();
        this.metrics = body;
        this.metricsFetchedAt = Date.now();
        this.metricsError = null;
      } catch (e) {
        this.metricsError = String(e);
      }
    },

    startMetricsTimer() {
      this.stopMetricsTimer();
      // Kick once immediately so the first sample shows up without a wait.
      this.refreshMetrics();
      this.metricsTimer = setInterval(() => this.refreshMetrics(), this.metricsAutoMs);
    },

    stopMetricsTimer() {
      if (this.metricsTimer) { clearInterval(this.metricsTimer); this.metricsTimer = null; }
    },

    onMetricsAutoToggle() {
      if (this.metricsAutoRefresh) this.startMetricsTimer();
      else this.stopMetricsTimer();
    },

    metricsAgeLabel() {
      if (!this.metrics || !this.metricsFetchedAt) return "";
      const secs = Math.max(0, Math.round((Date.now() - this.metricsFetchedAt) / 1000));
      return secs === 0 ? "just now" : `${secs}s ago`;
    },
  };
}
