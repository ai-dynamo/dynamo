# lib/http-server

`dynamo-http-server` is the plane-independent HTTP server for Dynamo services: the listener, write deadlines, service stages, connection monitor, and the `/metrics` scrape route. `dynamo-llm` serves the frontend through it. The frontend's request metrics are in `dynamo-metrics` ([`../metrics/AGENTS.md`](../metrics/AGENTS.md)).
