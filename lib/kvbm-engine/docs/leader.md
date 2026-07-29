# Leader Module

The leader module implements block coordination for a single KVBM instance. It owns
block metadata (via `BlockManager<G2>` and `BlockManager<G3>`), resolves cache lookups,
and orchestrates multi-stage onboarding sessions that move blocks between storage tiers
and across instances.

## Leader Trait

The `Leader` trait defines the core coordination interface:

```rust,ignore
pub trait Leader: Send + Sync {
    fn find_matches(&self, sequence_hashes: &[SequenceHash]) -> Result<FindMatchesResult>;
    fn find_matches_with_options(
        &self, sequence_hashes: &[SequenceHash], options: FindMatchesOptions,
    ) -> Result<FindMatchesResult>;
}
```

`find_matches` searches for blocks matching the given sequence hashes and returns
either an immediate result or an async session depending on the staging mode and
search scope.

## InstanceLeader

`InstanceLeader` is the primary implementation of `Leader`. It holds:
- `BlockManager<G2>` and optional `BlockManager<G3>` for local block registries
- A `ParallelWorkers` instance for driving transfer execution
- Session state for active onboarding operations
- Remote leader connections for cross-instance coordination

## FindMatchesResult

The result of `find_matches` is one of two variants:

- **`Ready`** -- Returned when `search_remote == false` AND `staging_mode == Hold`.
  Blocks are held in place via RAII without creating a session. The `ReadyResult`
  directly owns `Vec<ImmutableBlock<G2>>`.

- **`AsyncSession`** -- Returned when remote search or staging is required. Contains
  a `SessionId`, a `watch::Receiver<OnboardingStatus>` for progress tracking, and
  an optional `SessionHandle` for deferred control.

## StagingMode

Controls how matched blocks are staged and when the session completes:

| Mode | Behavior | Session Lifetime |
|------|----------|-----------------|
| `Hold` | Blocks remain in their current tiers (G2/G3) on original instances | Stays alive for deferred operations |
| `Prepare` | G3->G2 staging on all instances; no RDMA pulls | Stays alive after staging completes |
| `Full` | G3->G2 everywhere, then RDMA pull remote G2->local G2 | Completes when all blocks are in local G2 |

The progression `Hold -> Prepare -> Full` can be driven incrementally via
`SessionHandle::prepare()` and `SessionHandle::pull()`.

## OnboardingStatus State Machine

```text
Searching
    |
    +---> Holding { local_g2, local_g3, remote_g2, remote_g3, pending_g4, ... }
    |         |
    |         +---> (prepare) ---> Preparing { matched, staging_local, staging_remote }
    |                                  |
    +---> Preparing ------------------>+
    |                                  |
    |                            Prepared { local_g2, remote_g2 }
    |                                  |
    |                                  +---> (pull) ---> Staging { matched, ..., pulling }
    |                                                        |
    +---> Staging ------------------------------------------>+
                                                             |
                                                        Complete { matched_blocks }
```

Each status variant carries counters for progress tracking and cost analysis.
`Holding` includes G4 load tracking (`pending_g4`, `loaded_g4`, `failed_g4`).

## SessionHandle

`SessionHandle` provides deferred control over `Hold` and `Prepare` sessions:

- `prepare()` -- Trigger G3->G2 staging (Hold -> Prepare transition)
- `pull()` -- Trigger RDMA pull of remote G2->local G2 (Prepare -> Complete)
- `cancel()` -- Cancel session and release all held blocks

Not available for `StagingMode::Full` (which runs to completion automatically).

## BlockAccessor

`BlockAccessor` provides a stateless, `Send + Sync` interface for policy-based
block scanning. Each `find()` call independently searches G2 then G3, acquiring
blocks via RAII. The companion `PolicyContext` adds result collection via
`yield_item()` for streaming scan results back to the caller.
