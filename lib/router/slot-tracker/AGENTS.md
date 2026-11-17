# lib/router/slot-tracker

`dynamo-slot-tracker` is the slot tracker stage: it records the requests booked on each worker rank and reports their load and fullness. Hosts outside `lib/router` import through `host`.

`record_full_worker_ranks_at` fills a caller-owned `FullRankCounts` from the derived
prompt registry at one instant. It holds one read guard and creates no projection
map or per-request ownership state. The scheduler uses it when exclusive-affinity
costing or queued readiness needs current host-reject fullness.
