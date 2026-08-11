ARG SNAPSHOT_TOOLS_IMAGE
ARG SNAPSHOT_AGENT_IMAGE
FROM ${SNAPSHOT_AGENT_IMAGE} AS pinned-agent
FROM ${SNAPSHOT_TOOLS_IMAGE}

USER root
COPY --from=pinned-agent /usr/local/sbin/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
COPY --from=pinned-agent /usr/local/bin/cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
COPY --from=pinned-agent /usr/local/bin/nsrestore /usr/local/bin/nsrestore
COPY snapshot-entrypoint /usr/local/bin/snapshot-entrypoint
RUN chmod 0755 /usr/local/sbin/cuda-checkpoint \
    /usr/local/bin/cuda-checkpoint-helper \
    /usr/local/bin/nsrestore \
    /usr/local/bin/snapshot-entrypoint

ENTRYPOINT ["/usr/local/bin/snapshot-entrypoint"]
CMD []
