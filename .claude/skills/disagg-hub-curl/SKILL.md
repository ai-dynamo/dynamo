---
name: disagg-hub-curl
description: Reference card for the kvbm-hub HTTP surface — list instances, reset connector tiers, register remote leaders, and check health. Discovery port 1337, control port 8337.
---

# Skill: KVBM Hub HTTP Reference

Quick `curl` recipes against a running `kvbm_hub`. The hub mounts the
same routes on both ports; pick whichever is reachable.

## Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/v1/features/conditional-disagg/instances` | GET | List registered P/D participants by velo InstanceId |
| `/v1/instances/{id}/reset` | PUT | Reset block manager tiers on a specific connector (proxied via velo) |
| `/v1/instances/{id}/register_leader` | PUT | Tell connector `{id}` to register a remote leader for distributed search |
| `/v1/instances/{id}/health` | GET | Probe a connector's velo handler and report last-heartbeat info |
| `/v1/instances/{id}/heartbeat` | POST | Client-initiated TTL refresh (legacy; hub-push heartbeat handles this normally) |

The proxy routes (`reset`, `register_leader`, `health`) require the hub to
run with `--velo-port`; without it they return 503.

## List P/D split

```bash
curl -sS http://127.0.0.1:8337/v1/features/conditional-disagg/instances | jq
# {"prefill": ["uuid-..."], "decode": ["uuid-..."]}
```

## Reset

`tiers` is `Option<Vec<Tier>>`. `null`/omitted → "all available tiers,
skip what's not configured." Explicit list → fail atomically if any
listed tier is missing.

```bash
HUB=http://127.0.0.1:8337
PID=<prefill-instance-id>

# Reset everything that exists on this connector
curl -sS -X PUT $HUB/v1/instances/$PID/reset \
  -H 'content-type: application/json' -d '{}' | jq
# {"reset":["g2"], "failed":[], "skipped_unconfigured":["g3"]}

# Reset just G2 — succeeds if G2 is configured
curl -sS -X PUT $HUB/v1/instances/$PID/reset \
  -H 'content-type: application/json' -d '{"tiers":["g2"]}' | jq
# {"reset":["g2"], "failed":[], "skipped_unconfigured":[]}

# Reset G3 when not configured — fails atomically, G2 is NOT touched
curl -sS -X PUT $HUB/v1/instances/$PID/reset \
  -H 'content-type: application/json' -d '{"tiers":["g3"]}'
# 400 {"error":"TierNotConfigured(g3)"}
```

## Register a remote leader

```bash
DECODE_ID=<decode-instance-id>
PREFILL_ID=<prefill-instance-id>

# Tell the decode instance about the prefill instance
curl -sS -X PUT $HUB/v1/instances/$DECODE_ID/register_leader \
  -H 'content-type: application/json' \
  -d "{\"instance_id\":\"$PREFILL_ID\"}" | jq
# {"remote_leaders":[...]}
```

## Health probe

```bash
curl -sS $HUB/v1/instances/$PID/health | jq
# {"velo_reachable": true, "last_heartbeat_at_ms": 1777068200123, "consecutive_failures": 0}
```

## Direct connector access (opt-in)

When `KVBM_CONTROL_ENABLED=true` is set on the connector, the same
operations are available directly on `0.0.0.0:9999` (or whatever
`KVBM_CONTROL_PORT` is set to). Default is **off** — operators should
prefer the hub.

## Notes

- The hub's two ports today are functionally identical for these routes;
  the `velo_port` is separate and is for active-message traffic, not HTTP.
- All proxy responses preserve the connector's status code where it makes
  sense. Velo timeouts surface as 504, velo "instance not found" as 404.

## See also

- `/disagg-bringup` — bring up a hub + P + D before running these
- `/disagg-teardown` — clean up afterwards
