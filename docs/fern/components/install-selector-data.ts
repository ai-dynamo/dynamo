/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Install-selector view of the shared release data. Keep release and backend
 * versions in releases.data.ts; this module only formats install commands.
 */

import {
  CURRENT_VERSION,
  CURRENT_WHEEL,
  MAIN_TOT,
  RELEASES,
  type BackendPins,
} from "./releases.data";

export type InstallBackend = "sglang" | "trtllm" | "vllm";
export type InstallChannel = "stable" | "nightly" | "source";
export type InstallForm = "container" | "wheel";

export type InstallEntry = {
  backend_version: string;
  dynamo?: string;
  latest?: boolean;
  source?: boolean;
  note?: string;
  commands: Partial<Record<InstallForm, string>>;
};

export type InstallFramework = {
  label: string;
  stable: InstallEntry[];
  nightly: InstallEntry[];
  source: InstallEntry[];
};

export type InstallData = Record<InstallBackend, InstallFramework>;

type Backend = {
  id: InstallBackend;
  label: string;
  image: string;
  extra?: string;
};

const BACKENDS: Backend[] = [
  { id: "sglang", label: "SGLang", image: "sglang", extra: "sglang" },
  { id: "trtllm", label: "TensorRT-LLM", image: "tensorrtllm" },
  { id: "vllm", label: "vLLM", image: "vllm", extra: "vllm" },
];

function pin(pins: BackendPins | undefined, backend: InstallBackend): string | undefined {
  return pins?.[backend];
}

function dockerCommand(image: string, tag: string): string {
  return `docker run --gpus all --network host --ipc host --rm -it nvcr.io/nvidia/ai-dynamo/${image}:${tag}`;
}

function stableWheelCommand(backend: Backend, version: string): string {
  const wheel = version === CURRENT_VERSION.slice(1) ? CURRENT_WHEEL : version;
  const prerelease = backend.id === "sglang" ? "--prerelease=allow " : "";
  return `uv pip install ${prerelease}"ai-dynamo[${backend.extra}]==${wheel}"`;
}

function stableEntries(backend: Backend): InstallEntry[] {
  return RELEASES.filter(
    (release) =>
      (release.kind === "stable" || release.kind === "patch") &&
      pin(release.pins, backend.id),
  )
    .slice(0, 3)
    .map((release) => {
      const version = release.version.replace(/^v/, "");
      return {
        backend_version: pin(release.pins, backend.id)!,
        dynamo: version,
        commands: {
          container: dockerCommand(`${backend.image}-runtime`, version),
          ...(backend.extra
            ? { wheel: stableWheelCommand(backend, version) }
            : {}),
        },
      };
    });
}

function nightlyEntries(backend: Backend): InstallEntry[] {
  const backendVersion = pin(MAIN_TOT, backend.id);
  if (!backendVersion) return [];

  return [
    {
      backend_version: backendVersion,
      latest: true,
      commands: {
        container: dockerCommand(`${backend.image}-runtime-nightly`, "latest"),
      },
    },
  ];
}

function sourceEntries(backend: Backend): InstallEntry[] {
  if (backend.id === "trtllm") return [];

  const backendVersion = pin(MAIN_TOT, backend.id) ?? "main";
  const image = `dynamo:latest-${backend.image}-xpu-runtime`;
  const command = [
    "git clone https://github.com/ai-dynamo/dynamo.git",
    "cd dynamo",
    `container/render.py --framework=${backend.id} --device=xpu --target=runtime`,
    `docker build -t ${image} \\\n  -f container/${backend.image}-runtime-xpu-amd64-rendered.Dockerfile .`,
    `container/run.sh --image ${image} --device=xpu -it`,
  ].join("\n");

  return [
    {
      backend_version: backendVersion,
      source: true,
      note: "Intel XPU runtime images are built locally from the Dynamo repository.",
      commands: { container: command },
    },
  ];
}

export const INSTALL_DATA = Object.fromEntries(
  BACKENDS.map((backend) => [
    backend.id,
    {
      label: backend.label,
      stable: stableEntries(backend),
      nightly: nightlyEntries(backend),
      source: sourceEntries(backend),
    },
  ]),
) as InstallData;

export default INSTALL_DATA;
