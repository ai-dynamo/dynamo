FROM lmsysorg/sglang:v0.5.13.post1-cu130-runtime@sha256:50197bfb26d7cd4d434bfbd4308cb43088ddd7536ea8dcc54b669fee3900439a

ARG DYNAMO_COMMIT=unknown
ARG SGLANG_COMMIT=unknown
LABEL org.opencontainers.image.revision.dynamo=$DYNAMO_COMMIT
LABEL org.opencontainers.image.revision.sglang=$SGLANG_COMMIT
LABEL org.opencontainers.image.base.name="lmsysorg/sglang:v0.5.13.post1-cu130-runtime"

USER root
COPY *.whl /tmp/runtime-wheels/
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    /tmp/runtime-wheels/*.whl && \
    python3 -m pip install --no-cache-dir \
        "distro==1.9.0" "kubernetes>=32.0.1,<33.0.0" && \
    rm -rf /tmp/runtime-wheels

COPY dynamo/components/src /opt/dynamo/components/src
COPY sglang/python /opt/sglang/python
COPY verify_runtime.py /opt/decode-migration/verify_runtime.py
ENV PYTHONPATH=/opt/dynamo/components/src:/opt/sglang/python
RUN python3 /opt/decode-migration/verify_runtime.py
