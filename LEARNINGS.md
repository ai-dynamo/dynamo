# Learnings

## Checkpoint identity and reuse

- Dynamo checkpoint reuse is identity-hash based, not model-name based.
- The identity hash includes:
  - `model`
  - `backendFramework`
  - `dynamoVersion`
  - `tensorParallelSize`
  - `pipelineParallelSize`
  - `dtype`
  - `maxModelLen`
  - `extraParameters`
- A manual checkpoint for `Qwen/Qwen3-0.6B` with `dtype=bfloat16`, `maxModelLen=4096`, and `extraParameters.testCase=manual-cross-node` does not match an auto DGD identity that only specifies model/framework/TP/PP.
- This is why `vllm-qwen3-manual-a` had hash `d792c72d7306e944`, while the auto-created checkpoint had hash `73e74442beb109ed`.
- `vllm-qwen3-manual-a` is only the CR name. The real checkpoint identity is `status.identityHash`.
- If the intent is to reuse a known checkpoint, `checkpointRef` is the safest path.
- If the intent is to use identity-based reuse, the identity fields must match exactly. Test-only identity fields will prevent reuse.

## Auto checkpoint behavior

- Auto mode created a new `DynamoCheckpoint` because the identity did not match the existing manual checkpoint.
- In the observed `vllm-2w-auto` run, the auto-created checkpoint succeeded, but the worker deployment had already been created on the cold-start path.
- The worker pods cold-started because the initial worker deployment template did not have restore-target labels or the `sleep infinity` placeholder override.
- A later spec change to the DGD did trigger a new worker hash and a restore-target rollout.
- The important subtlety is that the old worker hash and the new worker hash can coexist during that transition.
- In practice, the DGD status moved to the manual checkpoint, a new worker DCD and Deployment were created for the new hash, but the old worker DCD and Deployment were still alive long enough to keep acting on the old auto checkpoint hash.
- That means the system can temporarily have:
  - old worker-hash pods restoring from the old auto-generated checkpoint
  - new worker-hash pods restoring from the manual `checkpointRef`
- This is not a logical conflict in the checkpoint resolver. It is a rollout-timing problem caused by patching the checkpoint source after one restore path is already in flight.
- Practical consequence: if a DGD is expected to restore from an already known checkpoint, use `checkpointRef` from the start instead of relying on identity-based auto discovery.

## Live patching a DGD checkpoint source is messy

- Patching a live worker from auto checkpointing to `checkpointRef` worked, but it created a mixed transition state.
- Old worker-hash pods that were already in flight kept restoring from the old checkpoint hash even after the DGD spec changed.
- Deleting the auto-generated `DynamoCheckpoint` after those pods already existed did not stop those already-labeled pods from attempting restore.
- The operator eventually completed the rolling update, but the transition was hard to reason about because restore operations are serialized by the snapshot agent and both worker hashes remained visible during the rollout.
- Operationally, this looks like an "auto mode vs checkpointRef conflict," but what actually happened was:
  - auto mode created and started using a checkpoint
  - the DGD was patched to a manual checkpoint reference
  - the operator created a new worker hash for the manual checkpoint path
  - already-created old-hash pods continued to consume the old auto checkpoint until that rollout drained
- The clean operational path is:
  - delete the mixed DGD
  - wait for old worker pods to disappear
  - recreate the DGD with the intended `checkpointRef`

## Snapshot restore slot behavior

- The snapshot agent serializes restores with a restore slot.
- On a node with multiple restore-target pods, later restore pods wait until earlier restores finish.
- In practice, stale old-hash restore pods delayed the second new-hash restore pod even though the DGD had already been switched to the manual checkpoint.
- This makes mixed rollouts slower and harder to reason about.

## Checkpoint PVC drift

- The checkpoint create path and the restore path are asymmetric when the `DynamoCheckpoint` CR and the PVC drift out of sync.
- Current checkpoint-create behavior is:
  - checkpoint mode no longer short-circuits on an existing `/checkpoints/<hash>` directory
  - if a new checkpoint for the same hash reaches the snapshot agent, the orchestrator stages under `tmp/<hash>` and then replaces the published directory
  - practical result: a stale folder on the PVC is overwritten by a fresh checkpoint run
- Current restore behavior is different:
  - the operator trusts a `DynamoCheckpoint` CR in `Ready` phase
  - restore-target pods are created from the CR's checkpoint hash even if the PVC artifact folder has been deleted
  - the snapshot agent checks `/checkpoints/<hash>` before restore and simply skips restore if the folder is missing
  - practical result: the placeholder restore pod can sit there not ready, and the DGD can stay pending/degraded without a clean restore failure
- If the checkpoint PVC itself is missing, the failure happens even earlier:
  - restore-target pods fail at volume mount / claim resolution time
  - the snapshot agent never gets a chance to restore anything
- Operationally, stale checkpoint folders are now recoverable by rerunning checkpoint creation, but missing checkpoint artifacts behind a `Ready` CR are still a restore-time wedge.

## vLLM restore findings

- Manual checkpoint and restore for vLLM worked.
- A clean recreated DGD using `checkpointRef: vllm-qwen3-manual-a` is the right way to exercise restore deterministically.
- `manage_generation` is not required for vLLM checkpoint mode.
- vLLM does not need explicit wake tags unless the caller explicitly requested them.
- `sleep()` / `wake_up()` default behavior is correct for checkpoint/restore.

## SGLang restore findings

- The original SGLang restore failure was not caused by Dynamo creating the request-plane TCP listener before checkpoint.
- The problematic pre-runtime TCP listeners came from SGLang's PyTorch distributed initialization, specifically Gloo / c10d sockets in the scheduler process.
- Without loopback forcing, those sockets bound to the pod IP and CRIU restore failed with `Cannot assign requested address` on a new pod.
- For checkpoint mode, forcing the following was sufficient to make the SGLang socket setup restore-safe:
  - `GLOO_SOCKET_IFNAME=lo`
  - `NCCL_SOCKET_IFNAME=lo`
  - `NCCL_CUMEM_ENABLE=0`
  - `NCCL_P2P_DISABLE=0`
  - `NCCL_NVLS_ENABLE=0`
  - `NCCL_IB_DISABLE=1`
- `NCCL_P2P_LEVEL` does not need to be forced. Leaving it unset lets NCCL choose the best topology cutoff.
- `NCCL_CUMEM_HOST_ENABLE` does not need to be overridden for this purpose.
- After forcing loopback for Gloo/NCCL bootstrap sockets, fresh SGLang checkpoints captured the previously bad listeners on `127.0.0.1` instead of the pod IP.

## CRIU TCP policy

- `tcpEstablished` and `tcpClose` are alternative policies, not complementary ones.
- Valid operational combinations are:
  - `tcpEstablished=true`, `tcpClose=false`: preserve established TCP connections
  - `tcpEstablished=false`, `tcpClose=true`: restore only listening sockets and close all other TCP sockets
- `tcpEstablished=true` and `tcpClose=true` is contradictory and should be rejected.
- Both vLLM and SGLang restored successfully after the loopback fix with:
  - `tcpEstablished=true`
  - `tcpClose=false`
- Earlier failures with connected TCP sockets were partly due to the live snapshot-agent image not actually plumbing the Helm TCP settings yet. Rebuilding the snapshot-agent fixed that.

## `/dev/net/tun`

- A restore failure with `tun: Unable to create tun` was observed on SGLang, but it was not a stable per-node property.
- A later scale-out to 5 replicas succeeded across both previously suspect nodes.
- Both host nodes and snapshot-agent pods had `/dev/net/tun`.
- The restored worker containers did not explicitly mount `/dev/net/tun`.
- No dedicated `/dev/net/tun` stub/mount has been added to the DGD restore pod spec yet.
- Current conclusion: the earlier `tun` error was intermittent or path-specific, not a permanent node capability split.

## Node pinning and pod storms

- Pinning a GPU worker with `nodeName` to a node that is already full can create a rapid stream of failed replacement pods.
- This was caused by:
  - a DGD pinned to a single node
  - scaling replicas above the free GPU capacity on that node
  - a normal Deployment/ReplicaSet continuing to retry failed pods
- Because `nodeName` bypasses the scheduler, kubelet/device-plugin admission failed directly with `UnexpectedAdmissionError` instead of leaving the pod `Pending`.
- This is how the namespace ended up with a very large number of short-lived failed pods instead of one obviously stuck pod.
- The specific reason it exploded so hard is:
  - the Deployment still desired the higher replica count
  - each attempted pod was sent directly to the full node because of `nodeName`
  - kubelet rejected it immediately for lack of GPU capacity
  - the ReplicaSet immediately tried again
- On a fast failure loop, that can snowball into hundreds or thousands of pods and events very quickly.
- This is not primarily a snapshot bug. It is the expected failure mode of a Deployment whose desired state cannot fit on the pinned node.
- The operator currently does not guard against impossible `nodeName`-pinned GPU scale-ups.

## Operational guidance

- Prefer `checkpointRef` for controlled checkpoint/restore experiments.
- Do not put test-case metadata in checkpoint identities unless different hashes are intentional.
- Do not patch checkpoint source mid-rollout unless there is a strong reason. Recreate the DGD instead.
- Avoid `nodeName` for scale-out experiments unless the node has verified free GPU capacity.
- When testing restore, verify actual restore-target behavior by checking for:
  - `nvidia.com/snapshot-is-restore-target=true`
  - `nvidia.com/snapshot-checkpoint-hash=<hash>`
  - placeholder command `sleep infinity`
  - pod events `RestoreRequested` and `RestoreSucceeded`
