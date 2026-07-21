# NIXL 1.3.0 LIBFABRIC backport

Dynamo release 1.3 uses the NIXL 1.3.0 packages supplied by the SGLang 0.5.14
runtime image. The accompanying patch backports the per-rail `FI_MORE` flush
fix from [ai-dynamo/nixl#1966](https://github.com/ai-dynamo/nixl/pull/1966).

The patch is limited to the NIXL LIBFABRIC plugin. Dynamo rebuilds and replaces
that plugin without replacing NIXL's core libraries or Python packages. Remove
the backport when the SGLang runtime consumes a NIXL release containing the
upstream fix.
