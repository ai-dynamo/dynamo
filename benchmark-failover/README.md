# benchmark-failover

Failover-resilience benchmark runs against dynamo deployments. Each run lives under its own date-stamped subdirectory.

## Runs

| Run | Date | Description |
|---|---|---|
| [`20260501T205312Z-cascade-3x`](20260501T205312Z-cascade-3x/) | 2026-05-01 / 02 | 3-worker cascade test (kill 3 workers 60 s apart at T+600/660/720 s, observe ~15 min recovery) at c=24. 3 setups: baseline / failover / failover+migration. |

See each run's own `README.md` for setup details, headline metrics, and charts.
