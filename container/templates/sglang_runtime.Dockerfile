##################################
########## Runtime Image #########
##################################

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

# cleanup unnecessary libs
RUN apt remove -y python3-apt &&\
    pip uninstall -y termplotlib

# This ARG is still utilized for SGLANG Version extraction
ARG RUNTIME_IMAGE_TAG
WORKDIR /workspace

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/

ENV PATH=/usr/local/bin/etcd:$PATH

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    # Non-recursive chown - only the directories themselves, not contents
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    # No chmod needed: umask 002 handles new files, COPY --chmod handles copied content
    # Set umask globally for all subsequent RUN commands (must be done as root before USER dynamo)
    # NOTE: Setting ENV UMASK=002 does NOT work - umask is a shell builtin, not an environment variable
    && mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh

USER dynamo
# Copy attribution files
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/

# Copy ffmpeg
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/; \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/; \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/lib/pkgconfig/; \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/; \
    true # in case ffmpeg not enabled

# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 benchmarks/ /workspace/benchmarks/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

ENV SGLANG_VERSION="${RUNTIME_IMAGE_TAG%%-*}"
RUN --mount=type=bind,source=.,target=/mnt/local_src \
    pip install --no-cache-dir --break-system-packages \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        sglang==${SGLANG_VERSION}

# Install common and test dependencies
RUN --mount=type=bind,source=.,target=/mnt/local_src \
    pip install --no-cache-dir --break-system-packages \
        --requirement /mnt/local_src/container/deps/requirements.txt \
        --requirement /mnt/local_src/container/deps/requirements.test.txt \
        sglang==${SGLANG_VERSION} && \
    cd /workspace/benchmarks && \
    pip install --break-system-packages --no-cache . && \
    # pip/uv bypasses umask when creating .egg-info files, but chmod -R is fast here (small directory)
    chmod -R g+w /workspace/benchmarks && \
    # Install NVIDIA packages that are needed for DeepEP to work properly
    # This is done in the upstream runtime image too, but we overrode these packages earlier
    pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps \
        nvidia-nccl-cu12==2.28.3 \
        nvidia-cudnn-cu12==9.16.0.29 \
        nvidia-cutlass-dsl==4.3.0

# Copy tests, deploy and components for CI with correct ownership
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 components/ /workspace/components/
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/

# Enable forceful shutdown of inflight requests
ENV SGLANG_FORCE_SHUTDOWN=1

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

# Our scripting assumes /workspace is where dynamo is located
# In order to maintain the ability to have sglang and dynamo
# in the same workspace, symlink /workspace to /sgl-workspace/dynamo
USER root

# Fix directory permissions: COPY --chmod only affects contents, not the directory itself
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

RUN ln -s /workspace /sgl-workspace/dynamo

USER dynamo
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

ENV PATH=/home/dynamo/.local/bin:$PATH
# PYTHONPATH for root user access to dynamo packages (NVBug 5762058)
ENV PYTHONPATH=/home/dynamo/.local/lib/python3.12/site-packages:$PYTHONPATH
