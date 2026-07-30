/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kubernetes quickstart image selector. It keeps the quickstart's release,
 * custom-build, and XPU paths in one copyable block.
 */
"use client";

import { useState } from "react";

import { CURRENT_TAG } from "./releases.data";
import { LOCAL_SELECTOR_CSS } from "./local-selector-styles";

type Hardware = "nvidia" | "intel";
type Backend = "sglang" | "trtllm" | "vllm";
type Build = "release" | "custom";

type Option<T extends string> = {
  id: T;
  label: string;
  sub?: string;
};

const HARDWARES: Option<Hardware>[] = [
  { id: "nvidia", label: "NVIDIA GPU", sub: "published images" },
  { id: "intel", label: "Intel XPU", sub: "source build" },
];

const BACKENDS: Option<Backend>[] = [
  { id: "sglang", label: "SGLang" },
  { id: "trtllm", label: "TensorRT-LLM" },
  { id: "vllm", label: "vLLM" },
];

const BUILDS: Option<Build>[] = [
  { id: "release", label: "Release image", sub: `Dynamo ${CURRENT_TAG}` },
  { id: "custom", label: "Custom image", sub: "build and push" },
];

const RUNTIME_IMAGES: Record<Backend, string> = {
  sglang: "sglang-runtime",
  trtllm: "tensorrtllm-runtime",
  vllm: "vllm-runtime",
};

const FRAMEWORK_ARGS: Record<Backend, string> = {
  sglang: "sglang",
  trtllm: "trtllm",
  vllm: "vllm",
};

function cudaVersionFlag(backend: Backend): string[] {
  return backend === "trtllm" ? ["  --cuda-version=13.1 \\"] : [];
}

function backendEnabled(hardware: Hardware, backend: Backend): boolean {
  return hardware === "nvidia" || backend === "vllm";
}

function buildEnabled(hardware: Hardware, build: Build): boolean {
  return hardware === "nvidia" || build === "custom";
}

function commandFor(hardware: Hardware, backend: Backend, build: Build): string {
  if (hardware === "intel") {
    return [
      'export XPU_IMAGE="registry.example.com/vllm-runtime-xpu:quickstart"',
      "",
      "python3 container/render.py \\",
      "  --framework=vllm \\",
      "  --device=xpu \\",
      "  --target=runtime \\",
      "  --output-short-filename",
      'docker build --tag "$XPU_IMAGE" --file container/rendered.Dockerfile .',
      'docker push "$XPU_IMAGE"',
    ].join("\n");
  }

  if (build === "release") {
    return [
      `export DYNAMO_VERSION=${CURRENT_TAG}`,
      'export PLANNER_IMAGE="nvcr.io/nvidia/ai-dynamo/dynamo-planner:${DYNAMO_VERSION}"',
      `export RUNTIME_IMAGE="nvcr.io/nvidia/ai-dynamo/${RUNTIME_IMAGES[backend]}:\${DYNAMO_VERSION}"`,
    ].join("\n");
  }

  return [
    `export RUNTIME_IMAGE="registry.example.com/${RUNTIME_IMAGES[backend]}:quickstart"`,
    'export PLANNER_IMAGE="nvcr.io/nvidia/ai-dynamo/dynamo-planner:' + CURRENT_TAG + '"',
    "",
    "python3 container/render.py \\",
    `  --framework=${FRAMEWORK_ARGS[backend]} \\`,
    "  --device=cuda \\",
    ...cudaVersionFlag(backend),
    "  --target=runtime \\",
    "  --output-short-filename",
    'docker build --tag "$RUNTIME_IMAGE" --file container/rendered.Dockerfile .',
    'docker push "$RUNTIME_IMAGE"',
  ].join("\n");
}

function ChoiceRow<T extends string>({
  label,
  options,
  selected,
  onSelect,
  isDisabled = () => false,
  disabledTitle,
}: {
  label: string;
  options: Option<T>[];
  selected: T;
  onSelect: (value: T) => void;
  isDisabled?: (value: T) => boolean;
  disabledTitle?: (option: Option<T>) => string;
}) {
  return (
    <div className="lqs-row">
      <span className="lqs-label">{label}</span>
      <div className="lqs-options" role="group" aria-label={label}>
        {options.map((option) => {
          const disabled = isDisabled(option.id);
          return (
            <button
              key={option.id}
              type="button"
              className="lqs-chip"
              aria-pressed={selected === option.id}
              disabled={disabled}
              title={disabled ? disabledTitle?.(option) : undefined}
              onClick={() => onSelect(option.id)}
            >
              {option.label}
              {option.sub && <span className="lqs-chip-sub">{option.sub}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function KubernetesContainerSelector() {
  const [hardware, setHardware] = useState<Hardware>("nvidia");
  const [backend, setBackend] = useState<Backend>("vllm");
  const [build, setBuild] = useState<Build>("release");
  const [copyLabel, setCopyLabel] = useState("Copy");

  const command = commandFor(hardware, backend, build);
  const backendLabel = BACKENDS.find((option) => option.id === backend)?.label ?? backend;
  const hardwareLabel = hardware === "nvidia" ? "NVIDIA GPU" : "Intel XPU";
  const buildLabel = hardware === "intel"
    ? "Source build"
    : build === "release"
      ? "Published release"
      : "Custom runtime";

  function chooseHardware(next: Hardware) {
    setHardware(next);
    if (next === "intel") {
      setBackend("vllm");
      setBuild("custom");
    } else {
      setBuild("release");
    }
  }

  async function copyCommand() {
    if (!navigator.clipboard) return;
    await navigator.clipboard.writeText(command);
    setCopyLabel("Copied!");
    window.setTimeout(() => setCopyLabel("Copy"), 1200);
  }

  return (
    <>
      <style>{LOCAL_SELECTOR_CSS}</style>
      <section className="lqs-panel" aria-label="Kubernetes container image selector">
        <div className="lqs-head">
          <h3>Choose your Kubernetes image path</h3>
          <p>Use the variables from this block in the deployment steps below.</p>
        </div>

        <ChoiceRow label="Hardware" options={HARDWARES} selected={hardware} onSelect={chooseHardware} />
        <ChoiceRow
          label="Backend"
          options={BACKENDS}
          selected={backend}
          onSelect={setBackend}
          isDisabled={(value) => !backendEnabled(hardware, value)}
          disabledTitle={(option) => `${option.label} does not currently have an Intel XPU quickstart path.`}
        />
        <ChoiceRow
          label="Container"
          options={BUILDS}
          selected={build}
          onSelect={setBuild}
          isDisabled={(value) => !buildEnabled(hardware, value)}
          disabledTitle={() => "Intel XPU quickstart images are built from source and pushed to your registry."}
        />

        <div className="lqs-output">
          <div className={`lqs-rec lqs-rec--${build === "release" ? "stable" : "source"}`}>
            <div className="lqs-eyebrow">Kubernetes quickstart</div>
            <div className="lqs-title">
              <span className="lqs-badge">{build === "release" ? "Use" : "Build"}</span>
              {buildLabel}
            </div>
            <div className="lqs-support">{hardwareLabel} · {backendLabel}</div>
          </div>
          <div className="lqs-command">
            <button type="button" className="lqs-copy" onClick={copyCommand}>
              {copyLabel}
            </button>
            <pre>{command}</pre>
          </div>
        </div>
      </section>
    </>
  );
}

export default KubernetesContainerSelector;
