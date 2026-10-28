# LPX placement contract fixtures

These fixtures are the shared contract for compiler-owned LPU placement units
(`config.lpuUnits` in `deployment.v1.capnp.bin`). GBuild writes the units. The
Dynamo operator, Nova, Hydra and Cyborg read them. Every consumer checks its
own view of a fixture against the same reviewed expectation.

## Layout

Each fixture directory holds:

- `deployment.v1.capnp.bin`: the deployment contract, with `lpuUnits`, the only
  LPU placement it carries.
- `execution.v1.capnp.bin`: the execution contract, when it is 2 MiB or less.
  The export tool omits a larger file and records the omission in the index
  `notes`. The GBuild test needs this file for its consistency checks.
- `expected-placement.json`: the reviewed placement. A person writes it from the
  placement rules and the decoded partition facts. Never generate it with the
  code under test, and never copy `lpuUnits` from the bytes into it.

`fixtures.json` lists every fixture with:

- `name`: the directory name.
- `kind`: `compiled` (a real GBuild compile) or `builder` (written by GBuild's
  manifest builder from partition facts, for a shape that no preset compiles).
  A `builder` execution gives every chip one placeholder `continue` stage, so
  runtime manifest readers read it as written. It has no GAS files, so no
  runtime can run it.
- `buildId` (compiled) or `generator` (builder): where the bytes come from.
  A compiled bundle lives at
  `gs://bkt-p-lpu-models-registry-us-central1-460c/<buildId>/`.
- `gbuildCommit`: the monorepo commit whose GBuild wrote the bytes.
- `files`: the sha256 of every file in the directory.

The compiled bundles predate the capnp v13 schema, in which `numLpuNodes`,
`selectedPropSyncChains` and each partition's `topologyMetadata` became
`obsoleteNumLpuNodes`, `obsoleteSelectedPropSyncChains` and
`obsoleteTopologyMetadata` with the same ordinals. Their bytes still carry those
fields. Every reader ignores them, so the bytes stay as compiled. The builder
fixtures come from a GBuild that no longer writes them; a decoder shows the
unwritten `obsoleteNumLpuNodes` as 0.

## expected-placement.json

| Field | Meaning |
| --- | --- |
| `architecture`, `compilationMode`, `devicesPerNode` | Facts of the bundle's deployed LPU partitions. |
| `localPartitions` | The DGD `lpx.experimental.localPartitions` of the scenario: `null`, `{"mode": "all"}` or `{"mode": "ids", "ids": [...]}`. |
| `units` | The expected `config.lpuUnits`, with wire field names. |
| `localPartitionIds` | Sorted local unit roots (`LPX_LOCAL_PARTITION_IDS`). |
| `lpr` | LPX pipeline request partitions and connectors. `null` means the operator must refuse the bundle; `refusal` then says why. |
| `agentCount`, `agentPods` | The per-pod view: Agent pod index, unit root, LPX partition, rank, group size and leader pod. |
| `cyborgEndpoints` | `lpx` only: the first Agent pod of each kept unit. |
| `novaStages` | `lpuOnly` only: the Agent pods of each stage, in Nova host order. |

## Vendoring rule

A consumer copies this whole directory to its own `testdata/lpx_placement/`
without changes. Its test fails when the sha256 of a vendored file differs from
`fixtures.json`. To update a vendored copy, copy the directory again from this
location. Do not edit files in a vendored copy.

## Refreshing fixtures

Run the export tool from the repository root. It writes the contract files and
`fixtures.json`. It never writes `expected-placement.json`.

The contracts do not carry the boundary topology dimensions of a packed chain.
A compiled fixture with a packed chain, or an LP30 chain with connectors, names
its preset in `COMPILED_FIXTURE_PRESETS` in `test_lpx_placement_fixtures.py`.
The test re-derives every compiled fixture's units with the current unit rules,
from the fixture's recorded partition facts and the topology extent that GBuild
resolves from each partition's topology. The contracts no longer carry the
selected chains, so the fixture's multi-partition units name them.

1. Compile the preset on the branch under test, for example
   `bake --agent r //Groq/GBuild:gbuild --profile k8s --preset <preset>.json`.
   The compile uploads the bundle to `dev/<branch>/<preset>/<revision>`.
2. Copy the bundle into its fixture directory:

   ```bash
   bake --agent r //Groq/GBuild/test/unit:export_lpx_placement_fixtures \
     --root "$PWD/Groq/GBuild/test/fixtures/lpx_placement" \
     download <name> <buildId> --gbuild-commit <monorepo commit>
   ```

3. Regenerate the builder fixtures:

   ```bash
   bake --agent r //Groq/GBuild/test/unit:export_lpx_placement_fixtures \
     --root "$PWD/Groq/GBuild/test/fixtures/lpx_placement" \
     generate --gbuild-commit <monorepo commit>
   ```

4. Review `expected-placement.json` by hand against the placement rules and
   the decoded bundle. To decode a bundle, run
   `bake --agent r //Groq/GBuild/gbuild/tools:inspect_contract <dir>/deployment.v1.capnp.bin`.
   When the bytes and the reviewed units disagree, find which one is wrong.
   Do not change the expectation only to match the bytes.
5. Update the hashes, then run the test:

   ```bash
   bake --agent r //Groq/GBuild/test/unit:export_lpx_placement_fixtures \
     --root "$PWD/Groq/GBuild/test/fixtures/lpx_placement" index
   bake --agent test //Groq/GBuild/test/unit:test_lpx_placement_fixtures
   ```

The test checks the hashes, compares each deployment's `lpuUnits` with the
reviewed `units`, runs the deployment and execution consistency checks, checks
that every LPU chip program has a stage, re-derives each compiled fixture's
units, and regenerates the builder fixtures to confirm that their bytes are
unchanged. A compiled fixture whose units no longer re-derive needs a new
compile, not an edited expectation.
