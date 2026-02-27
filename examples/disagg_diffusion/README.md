# Disaggregated Diffusion Inference POC

Split a monolithic diffusion pipeline (Text Encoder → Transformer → VAE) into
independent stages that can run on separate GPUs and scale independently.

Design doc: [docs/design/disaggregated_diffusion.md](../../docs/design/disaggregated_diffusion.md)

## Phases

### Phase 0: Offline Validation (no Dynamo)

Proves that diffusers supports split execution: encode, denoise, and VAE decode
can run independently with serialized intermediate tensors.

```bash
# Single GPU, ~24 GB VRAM for FLUX.1-schnell
python phase0_validate/validate_split.py \
    --model black-forest-labs/FLUX.1-schnell \
    --prompt "A photo of a cat sitting on a windowsill" \
    --output-dir /tmp/disagg_validate
```

### Phase 1: Dynamo Stage Workers

Three independent Dynamo workers, each loading only its model component:

```bash
# Terminal 1: Encoder Worker (loads CLIP + T5, ~12 GB)
python phase1_workers/encoder_worker.py --model black-forest-labs/FLUX.1-schnell

# Terminal 2: Denoiser Worker (loads Transformer, ~24 GB)
python phase1_workers/denoiser_worker.py --model black-forest-labs/FLUX.1-schnell

# Terminal 3: VAE Worker (loads VAE, ~1 GB)
python phase1_workers/vae_worker.py --model black-forest-labs/FLUX.1-schnell
```

### Phase 2: Orchestrator

Chains the three stage endpoints into an end-to-end generation pipeline:

```bash
python phase2_orchestrator/run_disagg.py \
    --prompt "A photo of a cat sitting on a windowsill" \
    --output /tmp/disagg_output.png
```

Or use the all-in-one launch script:

```bash
bash launch/run_all.sh black-forest-labs/FLUX.1-schnell "A photo of a cat"
```

## Supported Models

Any diffusers pipeline that exposes `encode_prompt()` and supports
`prompt_embeds` / `output_type="latent"`. Tested with:

- `black-forest-labs/FLUX.1-schnell` (recommended, 4 steps)
- `stabilityai/stable-diffusion-3.5-medium`
