# D=16 ComputeLab run metadata

- Date: 2026-07-07
- Slurm job: `3018026`
- Node: `r6515-0048`
- Measured code: `7d29b240c6632d78bf06b1e44ac0773f133a5e3b`
- Corrected analyzer: `e8b9dbfb5aba89e922c71bbdd9b3e123003ef6c7`
- Corpus SHA-256: `4056bc655ad04816545472cfdc9c877b727e221a141c413de4bd606d4e36da6c`
- Resident-image SHA-256: `97bf27f6fd8b0a9c8aef6a287841e53c74e23655783d14ea38aa10f522fd7c74`
- Trace SHA-256: `b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`
- CPU binding: logical CPUs `0-63`, one sibling per physical core
- NUMA policy: `interleave:0`
- Matrix: 90 capacity trials, 15 iso-throughput trials, and 4 memory processes
- Iso retry: not required
- Node-side checksum verification: 232 files passed
- Governor/turbo supplement: a non-login-shell sysfs probe immediately after the run found no exposed `scaling_governor`, AMD `cpufreq/boost`, or `amd_pstate/status` fields; Intel `no_turbo` was not applicable

The first matrix on measured commit `42c1ac9a46` was invalidated because CRTC completion-record harvesting occurred after actual drain but was included in `total_elapsed`. The measured-code fix records actual query/update drain timestamps before backend-specific metric harvesting. No v1 result is included here.

The raw logs and original node-path checksum manifest are retained in `/home/scratch.rupei_gpu/ckf-crtc-d16-3018026/artifacts/v2` until explicit cleanup. The compact trial JSON, corrected aggregates, corpus/hardware manifests, and checksums are committed in this directory.
