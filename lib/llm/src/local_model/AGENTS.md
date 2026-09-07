# Local Model Metadata

`ModelRuntimeConfig` is intended for facts resolved authoritatively after an
engine starts. `context_length` is the effective limit enforced by the running
engine; the static model maximum belongs in
`ModelDeploymentCard::architectural_max_context_length`. Tensor model
configuration is model metadata and belongs at the top level of the card.

Some existing fields do not yet follow this ownership boundary. In particular,
declarative model, frontend, routing, worker-placement, and deployment policy
should generally live in dedicated metadata rather than being added to runtime
config. Preserve current behavior when working in this area, and avoid adding
more statically known configuration to `ModelRuntimeConfig`.
