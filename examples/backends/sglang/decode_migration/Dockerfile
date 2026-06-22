FROM nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-dev.1-cuda13

ARG DYNAMO_COMMIT=unknown
ARG SGLANG_COMMIT=unknown
LABEL org.opencontainers.image.revision.dynamo=$DYNAMO_COMMIT
LABEL org.opencontainers.image.revision.sglang=$SGLANG_COMMIT

USER root
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    --extra-index-url https://flashinfer.ai/whl/cu130 \
    sglang-kernel==0.4.3 \
    flashinfer-python==0.6.12 \
    flashinfer-cubin==0.6.12 \
    flashinfer-jit-cache==0.6.12+cu130
COPY *.whl /tmp/runtime-wheels/
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    /tmp/runtime-wheels/*.whl && \
    rm -rf /tmp/runtime-wheels

COPY dynamo/components/src /opt/dynamo/components/src
COPY sglang/python /opt/sglang/python
ENV PYTHONPATH=/opt/dynamo/components/src:/opt/sglang/python
