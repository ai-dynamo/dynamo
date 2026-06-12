# MiniMax M3 Known Issues

Last validated: 2026-06-12

This note tracks known MiniMax M3 compatibility gaps: backend/framework
limitations and brief request-contract gaps versus the provider-verifier contract.

## Backend and Framework Limitations

- **Disaggregation does not work for TRTLLM.** The TRTLLM backend does not
  support disaggregated serving for MiniMax M3.

- **Images are not supported in TRTLLM.** Multimodal image inputs are not
  available on the TRTLLM path for MiniMax M3.

- **TRTLLM works only with BF16 weights.** The TRTLLM backend requires BF16
  weight formats; other precisions are not supported.

- **Encoder disaggregation is not supported in any framework.** Encoder
  disaggregation is not available across Dynamo backends for MiniMax M3.

## Request Contract Gaps

- Media option fields such as `detail`, `fps`, and image sizing options are not
  fully accepted by the Dynamo OpenAI request schema.

- System messages with multimodal content (images in `system` role) are rejected
  during request parsing.

- Some malformed media inputs return backend `500` errors instead of provider
  `400` client errors.

- Some provider-specific media validation semantics are not implemented yet.
