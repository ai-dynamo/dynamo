FROM nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-dev.1-cuda13

USER root
RUN python3 -m pip install --no-cache-dir --force-reinstall sglang-kernel==0.4.3
COPY *.whl /tmp/runtime-wheels/
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    /tmp/runtime-wheels/*.whl && \
    rm -rf /tmp/runtime-wheels
