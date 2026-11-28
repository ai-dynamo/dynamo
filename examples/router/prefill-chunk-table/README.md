# Partially cached prefill

A request with 6,144 input tokens and a 2,048-token prefix hit still needs 4,096 new tokens. The worker processes this suffix against the existing context. The `dynamo-prefill-chunk-table` policy adds a charge for each extra scheduler step on a previously observed chunk path. It keeps Dynamo's native cache, load and decode terms. It uses the prefix length reported by the indexer's device, host and disk tiers. When tier counts are absent it uses the router's estimated cached-token count. The geometry is a routing-time estimate of the prefix the engine will reuse.

Configure a prefill or aggregated role in the router policy YAML:

```yaml
worker_selection:
  prefill:
    type: dynamo-prefill-chunk-table
    parameters:
      profile_file: /var/run/dynamo/prefill-profile.json
```

A separate producer supplies a recent observation window from the same model and engine configuration. Each segment is a pair of newly computed tokens and starting context tokens. Include only ranks for the current worker instances. The example below gives one cold prefill in the input file `window.json`:

```json
{"model_name":"model","routing_group":"default","block_size":16,"valid_until_unix_ms":1893456000000,
 "ranks":[{"worker_id":1,"dp_rank":0,"step_cost_blocks":0.5,"samples":[
  {"q":6144,"c":0,"at_ns":100,"segments":[[2048,0],[2560,2048],[1536,4608]]}
 ]}]}
```

Choose `step_cost_blocks` by dividing the measured fixed per-step time on that rank by the reference time for one prefill block. Use the same reference across candidate ranks. The example value illustrates the units.

```sh
python3 build_profile.py window.json /var/run/dynamo/prefill-profile.json
```

Set `valid_until_unix_ms` to the end of the observation window’s validity in your deployment. The router checks model, routing group, block size and expiry before applying the profile.

The publisher keeps the 64 most recently observed unambiguous lengths per rank, with up to 32 steps per path. It atomically replaces the profile file. Remove ranks or the file when the underlying engine, model or chunk budget changes. A background reader refreshes the profile once per second. Selection uses a nonblocking read and keeps the native score while the profile is unavailable.

For the example path, a 2,048-token prefix leaves two steps, so the policy adds one fixed step cost, scaled by the native prefill weight. A 4,608-token prefix leaves one step and adds zero. An unseen 3,072-token boundary also contributes zero. A table miss retains all the native scoring terms. This policy changes scoring for an incoming request. The host continues to own bookings and in-flight load.
