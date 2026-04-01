FROM lmsysorg/sglang:v0.5.10rc0-cu130
ARG ARCH=arm64

# Install flashinfer with matching jit-cache.
# The base image ships a flashinfer version paired with sglang 0.5.10rc0.
# We pin to 0.6.7 and install the jit-cache from the flashinfer wheel index
# so all three packages (python, cubin, jit-cache) are consistent.
RUN pip install "flashinfer-python==0.6.7" "flashinfer-cubin==0.6.7" && \
    pip install "flashinfer-jit-cache==0.6.7" --index-url https://flashinfer.ai/whl/cu130

# Fix dynamo sglang imports for sglang 0.5.10:
# get_local_ip_auto / get_zmq_socket moved from sglang.srt.utils to
# sglang.srt.utils.network; msgpack is a new dependency for dynamo.common.
RUN pip install msgpack && \
    SITE=/usr/local/lib/python3.12/dist-packages && \
    sed -i 's|from sglang.srt.utils import get_local_ip_auto, get_zmq_socket, maybe_wrap_ipv6_address|from sglang.srt.utils.network import get_local_ip_auto, get_zmq_socket; maybe_wrap_ipv6_address = lambda x: x|' \
        $SITE/dynamo/sglang/publisher.py && \
    sed -i 's|from sglang.srt.utils import get_local_ip_auto|from sglang.srt.utils.network import get_local_ip_auto|' \
        $SITE/dynamo/sglang/register.py && \
    sed -i 's|from sglang.srt.utils import get_local_ip_auto|from sglang.srt.utils.network import get_local_ip_auto|' \
        $SITE/dynamo/sglang/request_handlers/handler_base.py

# Install etcd / etcdctl
RUN curl -fsSL "https://github.com/etcd-io/etcd/releases/download/v3.6.7/etcd-v3.6.7-linux-${ARCH}.tar.gz" | \
    tar xz -C /usr/bin --strip-components=1 \
        "etcd-v3.6.7-linux-${ARCH}/etcd" \
        "etcd-v3.6.7-linux-${ARCH}/etcdctl"

# Install nats-server
RUN curl -fsSL "https://github.com/nats-io/nats-server/releases/download/v2.12.4/nats-server-v2.12.4-linux-${ARCH}.tar.gz" | \
    tar xz -C /usr/bin --strip-components=1 \
        "nats-server-v2.12.4-linux-${ARCH}/nats-server"

# Install ai-dynamo (without [sglang] extra to avoid overriding existing sglang install)
RUN pip install ai-dynamo
