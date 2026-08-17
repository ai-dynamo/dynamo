FROM regolo-vllm-snapshot:kv-l1-20260814

COPY kv-sleep-l1/kv-l1-control /usr/local/bin/kv-l1-control
COPY ghost-kv/ghost-kv-entrypoint /usr/local/bin/ghost-kv-entrypoint
COPY ghost-kv/kv_only_sitecustomize.py /etc/python3.12/sitecustomize.py
RUN chmod 0755 /usr/local/bin/kv-l1-control /usr/local/bin/ghost-kv-entrypoint \
    && chmod 0644 /etc/python3.12/sitecustomize.py
