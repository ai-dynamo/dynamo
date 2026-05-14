# Dynamo Repository Guidance

## Logging

- Do not add `tracing::info!` logs solely for debugging, flaky-test diagnosis, or CI visibility. Use `tracing::debug!` for those breadcrumbs, and keep them scoped and high-signal.
