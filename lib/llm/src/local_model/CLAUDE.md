# Local Model Metadata

`ModelRuntimeConfig` is intended for facts resolved authoritatively after an
engine starts, such as effective limits, capacity, data-parallel placement, and
resolved service endpoints.

Some existing fields do not yet follow this ownership boundary. In particular,
declarative model, frontend, routing, worker-placement, and deployment policy
should generally live in dedicated metadata rather than being added to runtime
config. Preserve current behavior when working in this area, and avoid adding
more statically known configuration to `ModelRuntimeConfig`.
