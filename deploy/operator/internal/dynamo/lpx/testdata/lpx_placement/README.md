# LPX placement contract fixtures

These fixtures are the shared contract for compiler-owned LPU placement units
(`config.lpuUnits` in `deployment.v1.capnp.bin`). GBuild writes the units. The
Dynamo operator, Nova, Hydra and Cyborg read them. Every consumer checks its
own view of a fixture against the same reviewed expectation.

## Layout

Each fixture directory holds:

- `deployment.v1.capnp.bin`: the deployment contract, with `lpuUnits`.
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
- `buildId` (compiled) or `generator` (builder): where the bytes come from.
  A compiled bundle lives at
  `gs://bkt-p-lpu-models-registry-us-central1-460c/<buildId>/`.
- `gbuildCommit`: the monorepo commit whose GBuild wrote the bytes.
- `files`: the sha256 of every file in the directory.

## expected-placement.json

| Field | Meaning |
| --- | --- |
| `architecture`, `compilationMode`, `devicesPerNode` | Facts of the bundle's deployed LPU partitions. |
| `localPartitions` | The DGD `lpx.experimental.localPartitions` of the scenario: `null`, `{"mode": "all"}` or `{"mode": "ids", "ids": [...]}`. |
| `units` | The expected `config.lpuUnits`, with wire field names. |
| `localPartitionIds` | Sorted local unit roots (`LPX_LOCAL_PARTITION_IDS`). |
| `lpr` | LPX pipeline request partitions and connectors for the units path. `null` means the operator must refuse the bundle; `refusal` then says why. |
| `agentCount`, `agentPods` | The per-pod view: Agent pod index, unit root, LPX partition, rank, group size and leader pod. |
| `cyborgEndpoints` | `lpx` only: the first Agent pod of each kept unit. |
| `novaStages` | `lpuOnly` only: the Agent pods of each stage, in Nova host order. |
| `legacy` | The legacy operator projection of the same bundle and scenario, with units ignored: LPR, Agent count, `/configs` rows and Cyborg `lpu_servers`. `lpu_servers` uses scaling-group template `lpx` and Agent template `agt`. |

The units-path LPR and Agent count equal `legacy` except for these expected
differences:

- `f5c-lp20-lpu-nochain-2p`: the operator refuses an LPU-only bundle with more
  than one unit; the legacy operator accepts it.
- `f9b-lp30-lpu-chain-32c`: the HX connector lists the compiled load set (64
  pairs). The legacy operator uses the `(srcHosts - 1) * dpn + i -> i`
  convention (16 pairs).
- Every LP30 fixture: the units path sends `allocationMetadata: {}`; the legacy
  path sends the lyra envelope.

## Vendoring rule

A consumer copies this whole directory to its own `testdata/lpx_placement/`
without changes. Its test fails when the sha256 of a vendored file differs from
`fixtures.json`. To update a vendored copy, copy the directory again from this
location. Do not edit files in a vendored copy.

## Refreshing fixtures

Run the export tool from the repository root. It writes the contract files and
`fixtures.json`. It never writes `expected-placement.json`.

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
reviewed `units`, runs the deployment and execution consistency checks, and
regenerates the builder fixtures to confirm that their bytes are unchanged.
