<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom vision encoders for the aggregated `dynamo.vllm` worker

Run a **text-only LLM** in vLLM and plug in your **own** vision encoder (bespoke
ViT/projector weights, your own CUDA-graph capture, your own batched forward).
Dynamo intercepts image inputs, runs them through your encoder, and splices the
resulting image embeds into a mixed `EmbedsPrompt` fed to the text LM — no
separate encode worker, no NIXL transfer.

You implement one class, `VisionEncoderBackend`
(`dynamo.vllm.multimodal_utils.vision_encoder_backend`), and point the worker at
it:

```bash
python -m dynamo.vllm --model <text-LM> \
    --custom-encoder-class my_pkg.encoders.MyEncoder \
    --enable-multimodal --enable-prompt-embeds
```

The author contract (see the class docstring for the full surface):

- `build(model_id, device)` — load weights/tokenizer on the actor thread.
- `preprocess(raw) -> Preprocessed{item, cost, bucket_key}` — off-thread, CUDA-free.
- `forward_batch(items, target_bucket=None) -> list[torch.Tensor]` — actor thread;
  one `(n_visual_tokens, lm_hidden_dim)` tensor per image, in input order.
- `get_image_placeholder_token_id()` — Qwen-family encoders get this free by
  subclassing `QwenVisionEncoderBackend` (`qwen_vision_encoder.py`).

Worked examples in this directory: `hitchhikers_vision_encoder.py` (a fake encoder
that returns a fixed phrase's embeddings — a semantic smoke that answers "42") and
`qwen3vl_vit_encoder.py` (the real Qwen3-VL vision tower with a CUDA-graph bucket
ladder).

## Sending pre-computed embeddings (no ViT in Dynamo)

If you have already computed per-image embeddings on the client (e.g. CLIP/ViT
outputs) and only need a small **projector** (`embed_dim → lm_hidden_dim`), you do
**not** need Dynamo to run a vision tower. Send each embedding **inline** as a
safetensors `data:` URI on a normal `image_url` content part, and write a backend
whose `preprocess` decodes it and whose `forward_batch` projects it. No new Dynamo
request format is required today (a first-class `image_embeds` request field is
planned as a follow-up).

**Client — encode each image's embeddings** (`(n_tokens, embed_dim)` tensor):

```python
import base64, torch
from safetensors.torch import save as st_save

def encode_embeds_data_uri(embeds: torch.Tensor) -> str:  # embeds: (n_tokens, dim)
    blob = st_save({"embeds": embeds.contiguous().cpu()})
    return "data:application/x-dynamo-embeds;base64," + base64.b64encode(blob).decode()

# ... then send it as a normal image content part:
content = [
    {"type": "text", "text": "describe the product"},
    {"type": "image_url", "image_url": {"url": encode_embeds_data_uri(my_embeds)}},
]
```

**Server — your projector backend** (decode + project; Dynamo splices the result):

```python
import torch
from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
from examples.custom_encoder.qwen_vision_encoder import QwenVisionEncoderBackend
# decode_embeds_data_uri: inverse of the client snippet above (base64 -> safetensors load)

class MyProjector(QwenVisionEncoderBackend):
    image_token_id = 151655                              # hardcode your model's <|image_pad|>

    def build(self, model_id):                           # pick your own device
        self.device = "cuda"
        self.proj = load_my_projector().to(self.device)  # nn.Linear/MLP: dim -> hidden

    def preprocess(self, image_url):                     # off-thread, CUDA-free
        return Preprocessed(item=decode_embeds_data_uri(image_url))

    @torch.inference_mode()
    def forward_batch(self, items, target_bucket=None):  # actor thread, CPU out
        sizes = [t.shape[0] for t in items]
        x = torch.cat(items, dim=0).to(self.device)
        y = self.proj(x)                                 # -> (sum_tokens, hidden)
        return [p.detach().cpu() for p in y.split(sizes, dim=0)]
```

That's it — the embeddings ride the existing `image_url` channel, your projector
maps them to the LM hidden dim, and `build_mixed_embeds` splices them at the
placeholder positions before the prompt reaches vLLM (`--enable-prompt-embeds`).
A reference stub used by the test suite lives at
`tests/utils/embeds_passthrough_encoder.py` (it passes embeddings through
unchanged — your real backend projects them).
