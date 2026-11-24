# lib/http-server

`dynamo-http-server` is the plane-independent HTTP server for Dynamo services: the listener, write deadlines, service stages, connection monitor, and request metrics. `dynamo-llm` serves the frontend through it and adapts model cards and discovery into its metrics (`lib/llm/src/http/service/metrics.rs`).

## Frontend Serving-Load Metrics

The frontend's request metrics (`src/metrics.rs`) also feed the frame (`src/load.rs`) that `dynamo-llm` publishes for the Pool Relay's `SubscribeServingLoad` (`lib/llm/src/frontend_load.rs`), so `/metrics` and the Relay report the same numbers. Three are specific to that use and are experimental (see the Metrics Catalog):

- `dynamo_frontend_awaiting_first_token_requests{model}` (gauge): admitted requests with no output token yet. Requests producing output are `clamp_min(dynamo_frontend_active_requests - dynamo_frontend_awaiting_first_token_requests, 0)`; the frame derives that value (saturating) rather than exporting a second gauge.
- `dynamo_frontend_inflight_input_tokens{model, state}` (gauge, `state` is `awaiting_first_token` or `generating`): known prompt tokens of in-flight requests. Sum over `state` for the total.
- `dynamo_frontend_input_tokens_total{model}` (counter): prompt tokens of every admitted request, counted at tokenization. Use `rate()`.

Keep them consistent: one request transition updates every family through the request's shared `RequestMetrics` handle. Create a model's series when it registers, never from the publish path, which only reads. Before adding or changing a frontend metric, name its reader, check it is not a query over existing metrics or a sibling that should be a label, write its series count in the pull request, and add it to the Metrics Catalog.
