---
name: kvbm-hub-bringup
description: Reusable kvbm_hub launcher + shell helpers. Builds kvbm_hub + kvbmctl, starts a hub serving a chosen feature set (indexer / p2p / disagg) as the runtime source of truth, and exposes sourceable functions to wait for health and render a vLLM --kv-transfer-config from the live hub via kvbmctl. Consumed by the kvbm smoke skills; not a standalone test.
---

# Skill: kvbm-hub-bringup

The single place that knows how to **bring up `kvbm_hub`** and turn its
`GET /v1/config` aggregate into a connector config. Smoke skills (kvindex,
disagg, p2p, …) reuse this instead of each carrying their own hub launcher.

The hub is the runtime source of truth: it serves the features named by
`--features`, validates every registrant against its `primary` config
(`block_size` / `max_seq_len` / `block_layout`), and publishes `/v1/config` —
the aggregate the connector handshake and `kvbmctl` consume.

## Files

| File | Purpose |
|---|---|
| `start-hub.sh` | Foreground `kvbm_hub` launcher, fully parameterized by env. Builds both `kvbm_hub` + `kvbmctl`. Sizing-agnostic: the caller passes block_size / max_seq_len / g2 from its own profile. |
| `hub-lib.sh` | Sourceable helpers: `kvbm_hub_build`, `kvbm_hub_wait_health`, `kvbm_hub_render_vllm`. No side effects on source. |

## start-hub.sh

```bash
KVBM_HUB_FEATURES=indexer \
KVBM_HUB_BLOCK_SIZE=16 KVBM_HUB_MAX_SEQ_LEN=1024 KVBM_HUB_G2_MEMORY_GIB=2 \
  bash start-hub.sh /path/to/hub.log &      # foreground; background from caller
```

Key env (all optional; see the script header for the full list + defaults):

- `KVBM_HUB_FEATURES` — csv subset of `p2p,disagg,indexer`;
  empty = all supported (deps auto-added, e.g. CD pulls in P2P).
- `KVBM_HUB_BLOCK_SIZE` / `KVBM_HUB_MAX_SEQ_LEN` — the hub `primary` must-match
  values. **Set `MAX_SEQ_LEN` to the model's `max_model_len`** so kvbmctl renders
  a matching `--max-model-len` and the index is sized right.
- `KVBM_HUB_G2_MEMORY_GIB` / `KVBM_HUB_G2_BLOCKS` — advisory G2 sizing the hub
  seeds into the rendered connector `cache.host` (the hub has no G2 itself).
- `KVBM_HUB_PREFILL_VLLM_URL` + `_MODEL` — enable the CD prefill dispatcher.
- `KVBM_HUB_KVBM` — newline-separated `KEY.PATH=VALUE` entries; each becomes a
  `--kvbm` flag on the hub binary. Use this to seed common free-field overrides
  (e.g. tokio worker counts, nixl backends) into `base_config` so all connectors
  inherit them without each launcher repeating the same flags.
- `KVBM_HUB_KVBM_CONFIG` — JSON blob → `--kvbm-config`; applied before
  `KVBM_HUB_KVBM` entries. Authoritative hub fields (`block_layout`,
  `leader.hub.*`, `leader.disagg.role`) cannot be clobbered by either.
- Ports: `KVBM_HUB_{DISCOVERY,CONTROL,VELO}_PORT`.

## hub-lib.sh

```bash
. "$REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

# Wait for readiness (control-port /health), failing fast if the hub PID dies:
kvbm_hub_wait_health "$CTRL_PORT" 300 "$HUB_PID" "$HUB_LOG"

# Render vLLM connector args from the live hub, then consume with eval-array so
# the shell-quoted compact JSON stays a single argv element:
RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB_URL" indexer \
    --kv-connector-module-path "$MODULE_PATH") || exit 1
eval "KV_ARGS=( $RENDERED )"
exec python -m vllm.entrypoints.openai.api_server ... "${KV_ARGS[@]}"
```

`kvbm_hub_render_vllm` wraps `kvbmctl config vllm --hub <url> --features
<csv>` and passes extra args (`--role`, `--kv-connector-module-path`, repeated
`--kvbm`) straight through. The hub fills in `block_size` / `max_model_len` /
`block_layout` / `leader.hub` / advisory `cache.host`; the caller supplies only
free fields. Authoritative fields can't be clobbered by `--kvbm` overrides.

## Consumers

- `kvindex-smoke` — `start-hub.sh` wrapper sets `KVBM_HUB_FEATURES=indexer`,
  profile sizing, and `KVBM_HUB_KVBM` common overrides; `launch-instance.sh`
  renders via `kvbm_hub_render_vllm` with no per-launcher `--kvbm` flags.
- `disagg-bringup` — `start-hub.sh` wrapper sets `KVBM_HUB_FEATURES=disagg`,
  sizing, CD dispatcher URL/model, and the full `KVBM_HUB_KVBM` deployment-wide
  set (tokio workers, nixl backends, control dev+metrics, onboard mode);
  `launch-prefill.sh` / `launch-decode.sh` render via `kvbm_hub_render_vllm`
  passing only the per-instance `--role prefill|decode`.
