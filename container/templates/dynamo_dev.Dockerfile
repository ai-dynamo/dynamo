##############################################
########## Dev entrypoint image ##############
##############################################

FROM base AS dynamo_dev

ARG ENABLE_KVBM
ARG ARCH_ALT

# Application environment variables
ENV DYNAMO_HOME=/opt/dynamo \
    CARGO_TARGET_DIR=/opt/dynamo/target

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        # required for AIC perf files
        git \
        git-lfs \
        # rust build packages
        clang \
        libclang-dev \
        protobuf-compiler \
        # sudo for dev stage
        sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Add sudo privileges to dynamo user
    && echo "dynamo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dynamo \
    && chmod 0440 /etc/sudoers.d/dynamo

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache \
    && chown -R dynamo: /opt/dynamo /home/dynamo /workspace \
    && chmod -R g+w /opt/dynamo /home/dynamo/.cache /workspace

# Switch to dynamo user
USER dynamo
ENV HOME=/home/dynamo

# Create and activate virtual environment
ARG PYTHON_VERSION
RUN uv venv /opt/dynamo/venv --python $PYTHON_VERSION

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Install common and test dependencies
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    UV_GIT_LFS=1 uv pip install \
        --no-cache \
        --requirement /tmp/requirements.txt \
        --requirement /tmp/requirements.test.txt

# NIXL environment variables
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu/plugins
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${LD_LIBRARY_PATH}

# Copy ucx and nixl libs
COPY --chown=dynamo: --from=wheel_builder /usr/local/ucx/ /usr/local/ucx/
COPY --chown=dynamo: --from=wheel_builder ${NIXL_PREFIX}/ ${NIXL_PREFIX}/
COPY --chown=dynamo: --from=wheel_builder /opt/nvidia/nvda_nixl/lib64/. ${NIXL_LIB_DIR}/

# Copy built artifacts
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/
COPY --chown=dynamo: --from=wheel_builder $CARGO_TARGET_DIR $CARGO_TARGET_DIR
COPY --chown=dynamo: --from=wheel_builder $CARGO_HOME $CARGO_HOME

COPY --chown=dynamo: ./ /workspace/

RUN uv pip install \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
    /opt/dynamo/wheelhouse/nixl/nixl*.whl && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        uv pip install /opt/dynamo/wheelhouse/kvbm*.whl; \
    fi \
    && cd /workspace/benchmarks \
    && UV_GIT_LFS=1 uv pip install --no-cache .

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

# Setup environment for all users
USER root
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
