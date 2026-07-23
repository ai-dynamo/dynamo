# CMAF Binary Video Streaming

A minimal browser demo that plays a generated video as a continuous CMAF stream
over a single `POST /v1/videos/stream/binary/cmaf` response.

## How To Run

Start each of these (adjust hosts/ports/model for your setup):

1. **Frontend**

   ```bash
   python -m dynamo.frontend --router-mode round-robin --http-port 8085
   ```

2. **vLLM-Omni worker**

   ```bash
   export DYNAMO_VAAPI_DEVICE=/dev/dri/renderD128
   python -m dynamo.vllm.omni \
     --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
     --output-modalities video \
     --media-output-fs-url file:///tmp/ \
     --enforce-eager --vae-use-tiling --vae-use-slicing
   ```

3. **Same-origin proxy** (serves `client.html` and proxies the frontend under one
   origin, so no CORS changes are needed)

   ```bash
   python examples/custom_backend/cmaf_binary_video_streaming/run_proxy.py \
     --bind 0.0.0.0 --proxy-port 8086 --frontend-port 8085
   ```

Then open the client in a browser at the proxy URL, e.g.
`http://<your-host>:8086/`, and follow the on-page instructions: enter the model
and prompt, then press **Start CMAF Stream**.
