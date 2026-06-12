# MiniMax M3 Recipes

Last validated: 2026-06-12

Functional deployment recipes for MiniMax M3:

- [SGLang](sglang/)
- [vLLM](vllm/)
- [TRT-LLM](trtllm/)

## Table of Contents

- [Recipes](#recipes)
- [Performance Disclaimer](#performance-disclaimer)
- [Known Issues](#known-issues)
- [Backend and Framework Limitations](#backend-and-framework-limitations)
- [Request Contract Gaps](#request-contract-gaps)

## Recipes

Recipes are organized by backend, serving topology, and weight format. For
example, aggregated deployments live under paths such as
`<backend>/agg/<BF16-or-MXFP8>/deploy.yaml`.

## Performance Disclaimer

NOTE: Performance of MiniMax M3 is actively being tuned across SGLang, vLLM,
and TRT-LLM, so optimal flags and configs may vary as framework integrations
change.

## Known Issues

This section tracks known MiniMax M3 compatibility gaps: backend/framework
limitations and brief request-contract gaps versus the provider-verifier
contract.

### Backend and Framework Limitations

- **Disaggregation not supported for TRTLLM Backend.** The TRTLLM backend does
  not support disaggregated serving for MiniMax M3. Support coming soon.

- **Image/video inputs are not supported in the Dynamo TRTLLM backend.**
  Multimodal image/video inputs are not available on the Dynamo TRTLLM backend
  path for MiniMax M3. Support coming soon.

- **TRTLLM works only with BF16 weights.** The TRTLLM backend requires BF16
  weight formats; other precisions are not supported.

- **Encoder disaggregation is not supported in any framework.** Encoder
  disaggregation is not supported in any framework for MiniMax M3 yet, so it is
  not yet available across Dynamo backends.

### Request Contract Gaps

- Media option fields such as `detail`, `fps`, and image sizing options are not
  fully accepted by the Dynamo OpenAI request schema.

- System messages with multimodal content (images in `system` role) are rejected
  during request parsing.

- Some malformed media inputs return backend `500` errors instead of provider
  `400` client errors.

- Some provider-specific media validation semantics are not implemented yet.
