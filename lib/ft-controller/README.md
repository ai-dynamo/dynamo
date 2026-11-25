<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-ft-controller

The world controller for vLLM fault tolerance: one decider per vLLM
fault-tolerance world, running outside the GPU pods. It watches every rank's
front end through vLLM's own fault-tolerance HTTP API and, after a fault,
either recovers the world in place (a `retry` or `scale_down` round) or
escalates to the world's supervisor, which restarts the whole world.

This crate is phase 1 of the DEP: a standalone binary for one world, with a
decision core that later hosts (Engine Group, the DEP #18892 world
controller) can drive unchanged.

## How it works

Every `--poll-interval` the controller reads
`GET {base}/v1/fault_tolerance/status` from every rank, all ranks at once.
With external load balancing each front end manages exactly one engine. A
poll step waits for its slowest answer, so while a front end hangs the
controller polls once per `--call-timeout`.

1. **Form.** The controller acts only after every rank has answered one
   poll. Every answer's engine id must equal its `--rank` (in vLLM 0.34.0 the
   id is the global data-parallel rank, which a scale_down never
   renumbers); a mismatch means swapped or duplicated URLs and stops the
   controller with exit code 2. While ranks have not answered, it logs a
   warning every 30 s: the world is not protected until every rank answers.
2. **Detect.** A fault is suspected as soon as any rank is not healthy. A
   rank is *dead* when it reports `dead`, or when its status endpoint has
   failed for `--dead-after`.
3. **Start the clock.** The recovery deadline (`--recovery-deadline`) runs
   from the first survivor that reports anything but `healthy`, the moment
   vLLM's own recovery clock (`engine_recovery_timeout_sec`, 120 s) starts.
   While every survivor still reports healthy, for example because the
   controller lost contact with one rank, nothing escalates: the controller
   keeps waking survivors and, once this has lasted `--recovery-deadline`,
   logs an error every 30 s. (Survivors normally notice a dead peer only
   after a collective timeout, 30 s in vLLM 0.34.0, so the first 30 s or so
   of every incident look like this.) If the rank answers again before any
   round named it, it counts as a member again, and once every rank is
   healthy the controller goes back to watching.
4. **Wake.** An idle survivor only notices a dead peer when a request runs.
   Survivors still `healthy` `--wake-after` after the fault was suspected get
   one tiny completion (`POST {base}/v1/completions`, model from
   `GET {base}/v1/models`, `max_tokens` 1, short timeout, result ignored).
   The gap between wakes of one rank doubles up to `--wake-max-gap` and
   stays there while the fault lasts.
5. **Decide.** Once every survivor reports `unhealthy` (vLLM accepts orders
   from no other state): no dead ranks means `retry`; dead ranks mean
   `scale_down` of exactly those ranks.
6. **Send one round.** The round is recorded on disk, then the same
   instruction and request id go to every survivor at once
   (`POST {base}/v1/fault_tolerance/apply`), each exactly once.
7. **Judge.** Polling continues until every survivor is `healthy` again
   (recovered), the round fails in a way that allows another round (every
   survivor is `unhealthy` and idle again: an order was refused or failed,
   or no survivor took the round up within `--checkin-timeout` because the
   orders were lost), or something fatal happens. A round may run until the
   recovery deadline; `--round-timeout` caps it.
8. **Recovered:** the removed ranks leave the membership and the controller
   keeps watching. **Failed:** a new round with a new request id, within
   `--max-rounds` and the deadline. **Fatal:** escalate.

The controller escalates when no survivor is left (every rank dead or
unreachable), a survivor of a sent round dies (reports `dead` or stays
unreachable for `--dead-after`), the evidence conflicts (a rank that reported
`dead` answers again; a rank declared dead answers again after a round named
it; a round recovers some survivors and fails on others; a survivor works on,
or finished, a round this controller did not send), the round budget is
spent, the recovery deadline passes, or a round cannot be recorded (it is
then never sent).

## vLLM prerequisites

The GPU world must be set up as follows, or recovery cannot work:

- **EPLB with redundant experts for `scale_down`.** In vLLM 0.34.0 a worker
  refuses `scale_down` unless EPLB is on with `num_redundant_experts` > 0,
  which re-hosts the dead rank's experts. That refusal happens after
  check-in, where any scale_down failure is terminal: every survivor turns
  `dead`, so the controller escalates on every rank loss. A `retry` still
  works.
- **Fault tolerance, external load balancing, one API server.** Start every
  rank with `--enable-fault-tolerance`, external data-parallel load
  balancing (one front end per rank) and `--api-server-count 1`. Without
  them vLLM refuses to start or serves no `/v1/fault_tolerance/*` routes,
  and the controller waits in forming.
- **A store host the other ranks can reach.** `dp_master_ip` defaults to the
  host of the rank's `--rank` URL. That is wrong when the URL is a Service, a
  gateway or a port-forward: pass `--store-host R=<pod IP>` for every rank.
  A loopback URL host is refused unless `--store-host` names it explicitly
  (for single-node tests).
- **No API key.** The controller sends no API key. A rank started with
  `--api-key` answers 401, which the controller counts as unreachable, so it
  never forms (or counts the rank dead).
- **Separate store ports per world.** Each store rebuild takes the next port
  from `--store-port-base`. Worlds that share nodes need ranges that do not
  overlap.
- **Direct connections.** The controller ignores `HTTP_PROXY`, `HTTPS_PROXY`
  and `ALL_PROXY` and always connects to the ranks directly.

## Round rules (vLLM 0.34.0)

- One round is the same instruction and the same `request_id` sent to every
  surviving engine; all must check in within 10 s
  (`_ROUND_CHECKIN_TIMEOUT_SEC`).
- vLLM never deletes check-in keys, so a request id is never reused. Ids
  are `<world>-e<epoch>-r<seq>-<8 random hex>`: the epoch grows each time the
  controller starts, the sequence number with each round.
- Orders are accepted only by engines whose status is `unhealthy`. A refusal
  shows in the status: `ft_state` `failed`, the reason in `ft_error` (for
  example `status is HEALTHY`). A second order while one is in flight at the
  same front end is rejected without any change to the status, so the
  controller never sends one: it waits until the round has settled.
- The front end keeps `ft_state`, `last_ft_request_id` and `ft_error` only
  until the engine pushes a new status. After a successful round a survivor
  reports plain `healthy`.
- `retry` params: `{}`. With `--retry-abort` the controller adds
  `{"abort": true}`, which vLLM 0.34.0 ignores.
- `scale_down` params: `{"removed_dp_ranks": [...]}`. When the removed set
  includes the rank hosting the coordination store (the lowest live rank),
  the round also carries `dp_master_ip` (the store address of the new lowest
  surviving rank) and `dp_store_port`. vLLM 0.34.0 rebuilds the store, before
  check-in, only in such a round; it ignores a store address anywhere else.
- Once a round has carried `dp_master_ip` and `dp_store_port`, every later
  round carries them again, on a new port: survivors of a failed store round
  may point at different stores. Ports start at `--store-port-base` and each
  store round takes the next one.
- A check-in timeout or an invalid order (for example a missing store
  address) fails the round before the survivors vote, and they stay
  `unhealthy` with `ft_state` `failed`, so a new round may follow. A failure
  after check-in fails the vote: a `retry` leaves the survivors `unhealthy`,
  but a `scale_down` turns them `dead` (vLLM requires a full restart).
- vLLM never kills another rank's process. A survivor that is not healthy
  again `engine_recovery_timeout_sec` (120 s by default) after its own fault
  exits. Keep `--recovery-deadline` below that.
- A successful round takes at least the NIXL EP GPU kernel timeout (30 s by
  default in 0.34.0), because each worker first waits for its hung kernels; a
  `scale_down` then redistributes experts, possibly reloading them from
  disk. Measure a real round before setting `--round-timeout`.

## Usage

```bash
dynamo-ft-controller \
  --world dsr1-world-0 \
  --rank 0=http://10.0.0.5:8000 \
  --rank 1=http://10.0.0.6:8000 \
  --rank 2=http://10.0.0.7:8000 \
  --rank 3=http://10.0.0.8:8000 \
  --state-dir /var/lib/ft-controller \
  --escalate-cmd 'kubectl delete pod -l world={world}'
```

For a single-node test where every front end listens on 127.0.0.1, name the
store host explicitly: `--store-host 0=127.0.0.1 --store-host 1=127.0.0.1`
and so on.

| Flag | Default | Meaning |
|------|---------|---------|
| `--world NAME` | required | World name, used in request ids and state file names (`[A-Za-z0-9._-]`) |
| `--rank R=URL` | required, repeated | Front-end base URL of rank `R`, the rank's global data-parallel rank |
| `--store-host R=HOST` | the URL's host, if not loopback | Address rank `R` hosts the coordination store on (`dp_master_ip`) |
| `--initial-store-rank` | the lowest `--rank` | Rank hosting the store at start; must be the lowest `--rank` (vLLM's rule) |
| `--store-port-base` | `29650` | Port of the first store rebuild |
| `--poll-interval` | `500ms` | Time between status polls |
| `--call-timeout` | `2s` | Timeout of each status poll and order |
| `--dead-after` | `10s` | An unreachable rank counts as dead after this |
| `--wake-after` | `2s` | Wake survivors still healthy this long after the fault |
| `--wake-max-gap` | `30s` | Longest gap between two wakes of one rank |
| `--wake-timeout` | `2s` | Timeout of a wake request |
| `--checkin-timeout` | `10s` | vLLM's check-in timeout; a round no survivor took up after this is lost |
| `--max-rounds` | `3` | Rounds per incident before escalating |
| `--round-timeout` | none | Give up on a round after this; by default it may run until the recovery deadline |
| `--recovery-deadline` | `110s` | Escalate if not recovered this long after the first survivor fault |
| `--retry-abort` | off | Send `{"abort": true}` with retry rounds (vLLM 0.34.0 ignores it) |
| `--state-dir DIR` | required unless `--dry-run` | Epoch and last round record |
| `--escalate-cmd TEMPLATE` | none | Run through `sh -c` when escalating |
| `--escalate-timeout` | `30s` | Kill the escalation command after this |
| `--dry-run` | off | Log decisions; send nothing, run nothing, write nothing (see below) |
| `--log-format` | `text` | `text` or `json` logs on stderr; `RUST_LOG` sets the level |

Durations accept `500ms`, `10s`, `2m` and so on, up to one day.

## State, escalation and output

- **State directory.** `<world>.epoch.json` holds the epoch and
  `<world>.round.json` the last round record: `world`, `epoch`, `seq`,
  `request_id`, `instruction`, `params`, `removed`, `survivors`, `store`,
  `sent_at` and `not_after` (RFC 3339, UTC), `next_store_port`, and
  `escalated`. Each file is replaced atomically (temporary file, fsync,
  rename). The record is written before any order is sent; a round that
  cannot be recorded is not sent and the controller escalates.
- **Restarts.** A restarted controller starts a new epoch and continues the
  store port sequence after `next_store_port`, so ports never repeat while
  the record is kept. If the last round was not followed by an escalation,
  the world still runs with that round's survivors: `--rank` may list only
  those (anything else exits with code 2 and names the ranks), and if that
  round carried a store address, every later round carries one too. After an
  escalation (`escalated` is true) the supervisor restarts the whole world,
  so any `--rank` list is accepted and the store is back on the lowest rank.
  To protect a world that was restarted some other way, delete
  `<world>.round.json` first; the epoch file keeps request ids unique.
- **Escalation fails closed.** The controller logs at error level, marks the
  last round record `escalated`, runs `--escalate-cmd` (`{world}` and
  `{reason}` become single-quoted shell words; `FT_CONTROLLER_WORLD` and
  `FT_CONTROLLER_REASON` are also set), prints
  `FT_CONTROLLER_ESCALATED world=... reason=...` and exits with code 3, so a
  supervisor restarts the world. The reason is always one line: line breaks
  in vLLM errors are escaped.
- **Dry run.** Where it would escalate, a dry run prints
  `FT_CONTROLLER_WOULD_ESCALATE world=... reason=...`, does not exit, and
  watches again once every rank is healthy. As it sends no wakes, idle
  survivors never notice a fault, so a dry run often ends on the deadline.
- **Exit codes.** 0 after SIGINT or SIGTERM, 1 when the controller itself
  fails (for example an unreadable state directory), 2 for bad arguments or
  a world that contradicts them (an engine id that is not its rank, a
  `--rank` the last round removed), 3 after an escalation. A dry run never
  exits with code 3.
- **Recovery summary.** After each recovery the controller prints one line
  on stdout:

  ```text
  FT_CONTROLLER_RECOVERED world=w instruction=scale_down removed=[2] survivors=[0,1,3] rounds=1 detect_s=2.503 round_s=31.204 total_s=33.707 request_id=w-e1-r1-1a2b3c4d
  ```

  `detect_s` runs from the first fault evidence to the first round sent,
  `round_s` from the successful round to every survivor healthy, and
  `total_s` from the first evidence to recovered.
- **Logs** are structured `tracing` events on stderr, inside an
  `ft_controller{world=...}` span, with a `phase` field (`forming`, `watch`,
  `detect`, `wake`, `decide`, `round`, `recovered`, `escalate`), the request
  id and elapsed times. Rank state changes are logged once per change.

## Design

| Module | Contents |
|--------|----------|
| `decision` | The host-neutral decision core: `decide` turns per-rank observations into `Wait`, `Wake`, `Cleared`, `Retry`, `ScaleDown` or `Escalate`; `judge_round` judges a round in flight; `check_identity` checks an engine id. No I/O, no clock reads |
| `engine` | `EngineFt` (status, apply, wake) and its HTTP implementation |
| `round` | `RoundStore` (epoch and round records) with file and memory implementations; request ids |
| `escalate` | `Escalator` and the `--escalate-cmd` implementation |
| `controller` | The loop that polls, decides and acts through those traits |
| `protocol`, `cli` | vLLM's wire types; the binary's arguments |

The I/O sits behind the three traits so the same core can run inside Engine
Group or the DEP #18892 world controller, and so tests can run it without
GPUs.

## Phase 1 limits

- One world per process, and a single controller replica: there is no lease
  yet, so two controllers for one world would race.
- Static membership from the command line, checked against the last round
  record at start (see Restarts).
- External load balancing only: each front end must manage one engine.
- No API key or client certificate support.
- The controller decides and orders; it never kills a process. A rank that
  is removed but still alive is left to vLLM and the supervisor.

## Later phases

- Membership from discovery instead of flags.
- A lease with epoch fencing, for highly available controller replicas.
- Sidecar agents next to each rank, for evidence beyond the HTTP status.
- Rendering the controller and its world from Kubernetes resources.

## Tests

```bash
cargo test -p dynamo-ft-controller
```

Unit tests cover the decision core, the wire types, the state files, the
escalation command and the arguments. Integration tests (`tests/`) run the
controller over real HTTP against a fake vLLM front end per rank (axum). The
fake follows vLLM 0.34.0: the status machine, the check-in rule (scaled down
from 10 s), failures before the vote that leave survivors `unhealthy`, status
pushes that replace the front end's entry, the rejection of a second order
in flight, store rebuilds, idle engines, lost orders and lost contact. It
records every controller mistake it can see. `tests/binary.rs` runs the
binary itself. Timeouts are scaled down to milliseconds, so the whole suite
runs in a few seconds.
