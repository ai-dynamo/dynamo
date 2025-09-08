# Dynamo frontend node.

Usage: `python -m dynamo.frontend [--http-port 8000]`.

This runs an OpenAI compliant HTTP server, a pre-processor, and a router in a single process. Engines / workers are auto-discovered when they call `register_llm`.

Requires `etcd` and `nats-server -js`.

This is the same as `dynamo-run in=http out=dyn`.
