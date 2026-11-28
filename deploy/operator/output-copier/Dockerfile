# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM docker.io/library/busybox:1.37.0-musl@sha256:f6a95d611e0a8780fabdb2aec43e31275b2720dfefa3e41b16835bfc42454478 AS tools

# The upstream kubectl image has no shell or text-processing utilities.
RUN mkdir /copier-bin && \
    for tool in sh date grep awk sed tr cat sleep; do \
        ln -s /bin/busybox /copier-bin/"${tool}"; \
    done

FROM registry.k8s.io/kubectl:v1.37.2@sha256:fe049faa65ae483ac200009a949a68fa4c9985daea12c9cb1195c9c7965301fc

COPY --from=tools /bin/busybox /bin/busybox
COPY --from=tools /copier-bin/ /bin/
ENV PATH=/bin:/usr/local/bin:/usr/bin HOME=/tmp
WORKDIR /tmp
USER 1000:1000
ENTRYPOINT []
CMD ["/bin/sh"]
