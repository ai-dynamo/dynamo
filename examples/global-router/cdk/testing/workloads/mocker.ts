// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { pinnedImage, positive } from "../../lib/config";

export interface MockerWorkload {
  readonly image: string;
  readonly model: string;
  readonly workers: number;
  readonly speedup: number;
  readonly computeGroup?: string;
  readonly runtimeVersionOverride?: string;
}

/** Only the workload differs: PoolStack installs this DGD exactly as it does a GPU recipe. */
export function mockerDgdSpec(config: MockerWorkload): Record<string, unknown> {
  pinnedImage(config.image); positive(config.workers, "workers");
  if (!config.model?.trim() || !Number.isFinite(config.speedup) || config.speedup <= 0) {
    throw new Error("Mocker needs a model and positive speedup");
  }
  const env = [
    { name: "DYN_SERVING_LOAD_PUBLISH", value: "1" },
    { name: "DYN_DISCOVERY_BACKEND", value: "kubernetes" },
    { name: "DYN_KUBE_DISCOVERY_MODE", value: "pod" },
    { name: "DYN_EVENT_PLANE", value: "zmq" },
    { name: "DYN_REQUEST_PLANE", value: "tcp" },
  ];
  const component = (name: string, type: string, replicas: number, args: string[]) => ({
    name, type, replicas,
    ...(config.runtimeVersionOverride ? { runtimeVersionOverride: config.runtimeVersionOverride } : {}),
    podTemplate: { spec: {
      ...(config.computeGroup ? { nodeSelector: { "dynamo.nvidia.com/compute-group": config.computeGroup } } : {}),
      containers: [{ name: "main", image: config.image, command: ["python3"], args, env,
        resources: { requests: { cpu: "250m", memory: "512Mi" }, limits: { memory: "2Gi" } } }],
    } },
  });
  return { components: [
    component("Frontend", "frontend", 1, ["-m", "dynamo.frontend", "--router-mode", "round-robin"]),
    component("decode", "decode", config.workers, ["-m", "dynamo.mocker", "--model-path", config.model,
      "--model-name", config.model, "--num-workers", "1", "--speedup-ratio", String(config.speedup)]),
  ] };
}
