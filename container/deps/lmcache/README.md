Place an optional LMCache patch at:

    container/deps/lmcache/local.patch

The vLLM container build applies this patch after cloning LMCache and before
building/installing LMCache from source. Pass a changing `LMCACHE_PATCH_SHA`
build arg when the patch changes so Docker does not reuse the cached framework
install layer.
