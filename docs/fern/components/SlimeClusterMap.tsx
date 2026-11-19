/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SlimeClusterMap — where each part of the Slime tutorial runs on a
 * Kubernetes cluster, and what the overlapping terms mean.
 *
 * The diagram shows the control plane, the operators that Helm installs once
 * per cluster, a node without GPUs, a GPU node, and the shared volume. Each
 * named box is a Pod, and each row inside a Pod is a process with a CPU or
 * GPU chip, so a reader can see that Slime and the Dynamo frontend run on CPU
 * while the trainer and each engine own a GPU.
 *
 * The term buttons above the diagram highlight what each word covers:
 * Kubernetes cluster, node, Pod, operator, DynamoGraphDeployment, RayJob, Ray
 * cluster, and Ray job. Hover previews a term; a click or tap pins it.
 *
 * Its Pods follow the Slime tutorial pages: one frontend and two SGLang worker
 * Pods from the DGD, and a single-Pod Ray cluster plus the submitter from the
 * RayJob. Keep them in step with the tutorial.
 *
 * WHY "use client": it keeps the selected term. The server renders the
 * diagram with no term selected.
 *
 * The CSS lives in SlimeClusterMapStyles.tsx, where check_style_components.py
 * guards it. The page-usage example is in README.md.
 */
"use client";

import { type ReactNode, useState } from "react";

import { SlimeClusterMapStyles } from "./SlimeClusterMapStyles";

type Lens = "cluster" | "node" | "pod" | "operator" | "dgd" | "rayjob" | "raycluster" | "rayjobscript";
type Group = "kubernetes" | "dynamo" | "ray";
type Owner = "dynamo" | "kuberay";

type Proc = { id: string; name: string; on: "CPU" | "GPU"; role?: "training" | "inference" };

type Pod = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  owner: Owner;
  name: string;
  /** Shown as the box's native tooltip. */
  detail: string;
  procs: readonly Proc[];
};

const PODS: readonly Pod[] = [
  {
    id: "op-dynamo",
    x: 310,
    y: 94,
    w: 222,
    h: 46,
    owner: "dynamo",
    name: "dynamo-operator",
    detail: "Installed with the dynamo-platform Helm chart. Creates Pods for DynamoGraphDeployments.",
    procs: [{ id: "proc-op-dynamo", name: "Dynamo operator", on: "CPU" }],
  },
  {
    id: "op-kuberay",
    x: 544,
    y: 94,
    w: 222,
    h: 46,
    owner: "kuberay",
    name: "kuberay-operator",
    detail: "Installed with the kuberay-operator Helm chart. Creates Ray clusters and submitter Pods for RayJobs.",
    procs: [{ id: "proc-op-kuberay", name: "KubeRay operator", on: "CPU" }],
  },
  {
    id: "frontend",
    x: 40,
    y: 196,
    w: 194,
    h: 46,
    owner: "dynamo",
    name: "frontend",
    detail: "dynamo-frontend image. Serves rollout requests on port 8000. Requests no GPU.",
    procs: [{ id: "proc-frontend", name: "Dynamo frontend", on: "CPU" }],
  },
  {
    id: "submitter",
    x: 40,
    y: 250,
    w: 194,
    h: 46,
    owner: "kuberay",
    name: "submitter",
    detail: "A Kubernetes Job that runs ray job submit with the entrypoint and streams the driver log. Requests no GPU.",
    procs: [{ id: "proc-submit", name: "ray job submit", on: "CPU" }],
  },
  {
    id: "head",
    x: 282,
    y: 196,
    w: 151,
    h: 68,
    owner: "kuberay",
    name: "Ray head",
    detail: "Slime image. Requests 1 GPU, 16 CPUs, and 200 GiB of memory, and mounts /data.",
    procs: [
      { id: "proc-slime", name: "Slime driver", on: "CPU" },
      { id: "proc-megatron", name: "Megatron-LM", on: "GPU", role: "training" },
    ],
  },
  {
    id: "w0",
    x: 445,
    y: 196,
    w: 151,
    h: 68,
    owner: "dynamo",
    name: "worker-0",
    detail: "sglang-runtime image. Requests 1 GPU, serves the SGLang API on port 30000, and mounts /data.",
    procs: [
      { id: "proc-sglang0", name: "SGLang engine", on: "GPU", role: "inference" },
      { id: "proc-sidecar0", name: "Dynamo sidecar", on: "CPU" },
    ],
  },
  {
    id: "w1",
    x: 608,
    y: 196,
    w: 151,
    h: 68,
    owner: "dynamo",
    name: "worker-1",
    detail: "sglang-runtime image. Requests 1 GPU, serves the SGLang API on port 30000, and mounts /data.",
    procs: [
      { id: "proc-sglang1", name: "SGLang engine", on: "GPU", role: "inference" },
      { id: "proc-sidecar1", name: "Dynamo sidecar", on: "CPU" },
    ],
  },
];

const POD_IDS = PODS.map((pod) => pod.id);

/** The objects that the tutorial applies, as the API server stores them. */
const OBJECTS = [
  { id: "obj-dgd", text: "DGD" },
  { id: "obj-rayjob", text: "RayJob" },
  { id: "obj-pvc", text: "PVC" },
] as const;

/** GPU slots on the GPU node: the trainer's, the two engines', and the free ones. */
const GPU_SLOTS = ["training", "inference", "inference", "free", "free", "free", "free", "free"] as const;

const LENSES: ReadonlyArray<{
  id: Lens;
  group: Group;
  label: string;
  lit: readonly string[];
  /** Kept at full strength although not lit, because they hold lit parts. */
  keep?: readonly string[];
  /** Dim every other Pod. */
  dim: boolean;
  text: ReactNode;
}> = [
  {
    id: "cluster",
    group: "kubernetes",
    label: "Kubernetes cluster",
    lit: ["frame"],
    dim: false,
    text: "All the machines that Kubernetes manages, called nodes, plus the control plane that stores every object you apply. Everything in the tutorial runs in one cluster.",
  },
  {
    id: "node",
    group: "kubernetes",
    label: "Node",
    lit: ["node-cpu", "node-gpu"],
    dim: false,
    text: "One machine in the cluster, usually a VM. The tutorial needs three free GPUs. A Pod that needs no GPU, such as the frontend, can run on any node; the scheduler picks one.",
  },
  {
    id: "pod",
    group: "kubernetes",
    label: "Pod",
    lit: POD_IDS,
    dim: false,
    text: "One or more containers that Kubernetes runs together on one node. Each named box here is a Pod. A Pod that requests a GPU gets that GPU to itself.",
  },
  {
    id: "operator",
    group: "kubernetes",
    label: "Operator",
    lit: ["op-dynamo", "op-kuberay", "addons", "watch"],
    dim: true,
    text: "A controller that watches the API server for one kind of object and creates Pods to match it. You install the Dynamo platform and KubeRay once per cluster with Helm, and each operator then acts on objects in every namespace.",
  },
  {
    id: "dgd",
    group: "dynamo",
    label: "DynamoGraphDeployment",
    lit: ["frontend", "w0", "w1", "obj-dgd"],
    dim: true,
    text: "The object that you apply in Serve Inference. The Dynamo operator creates the frontend Pod and one Pod per SGLang engine from it, and keeps them running until you delete it.",
  },
  {
    id: "rayjob",
    group: "ray",
    label: "RayJob",
    lit: ["head", "submitter", "obj-rayjob"],
    dim: true,
    text: "The object that you apply in Train. KubeRay creates a Ray cluster for it, plus a submitter Pod that submits the job and streams its log, and deletes the Ray cluster when the job ends.",
  },
  {
    id: "raycluster",
    group: "ray",
    label: "Ray cluster",
    lit: ["head"],
    dim: true,
    text: "The Pods that KubeRay creates to run Ray: only the head Pod here, plus worker Pods in bigger jobs. Ray calls each of these Pods a node, but each one is a Pod, not a machine.",
  },
  {
    id: "rayjobscript",
    group: "ray",
    label: "Ray job",
    lit: ["proc-slime", "proc-megatron"],
    keep: ["head"],
    dim: true,
    text: "The program that runs on the Ray cluster: Slime's training script and the Ray actors that it starts, such as the Megatron-LM trainer. The RayJob object is how you ask KubeRay to run it.",
  },
];

const GROUPS: ReadonlyArray<{ id: Group; label: string }> = [
  { id: "kubernetes", label: "Kubernetes" },
  { id: "dynamo", label: "Dynamo" },
  { id: "ray", label: "Ray" },
];

const DEFAULT_TEXT =
  "Each named box is a Pod. A GPU chip marks a process with a GPU of its own: the trainer and each engine get one, and everything else runs on CPU. Pick a term to see what it covers.";

const chipWidth = (text: string) => Math.round(text.length * 6.3 + 14);

export function SlimeClusterMap() {
  const [pinned, setPinned] = useState<Lens | null>(null);
  const [hovered, setHovered] = useState<Lens | null>(null);
  const active = LENSES.find((lens) => lens.id === (hovered ?? pinned)) ?? null;

  const state = (id: string) => {
    if (!active) return "";
    if (active.lit.includes(id)) return " is-lit";
    if (active.keep?.includes(id)) return "";
    return active.dim && (POD_IDS.includes(id) || id.startsWith("obj-")) ? " is-dim" : "";
  };

  return (
    <div className={`slclu not-prose${active ? ` slclu--${active.group}` : ""}`}>
      <SlimeClusterMapStyles />

      <div className="slclu-terms" role="group" aria-label="Terms" onMouseLeave={() => setHovered(null)}>
        {GROUPS.map((group) => (
          <div key={group.id} className={`slclu-terms__group slclu-terms__group--${group.id}`}>
            <span className="slclu-terms__label">{group.label}</span>
            {LENSES.filter((lens) => lens.group === group.id).map((lens) => (
              <button
                key={lens.id}
                type="button"
                className="slclu-term"
                aria-pressed={pinned === lens.id}
                onMouseEnter={() => setHovered(lens.id)}
                onFocus={() => setHovered(lens.id)}
                onBlur={() => setHovered(null)}
                onClick={() => setPinned(pinned === lens.id ? null : lens.id)}
              >
                {lens.label}
              </button>
            ))}
          </div>
        ))}
      </div>

      <div className="slclu-stage">
        <svg
          className="slclu-svg"
          viewBox="0 0 800 414"
          role="img"
          aria-label="A Kubernetes cluster. The control plane stores the objects you apply, and the Dynamo and KubeRay operators, installed once with Helm, watch it. A node without GPUs runs the Dynamo frontend Pod and the RayJob's submitter Pod. A GPU node runs the Ray head Pod, with the Slime driver on CPU and Megatron-LM on one GPU, and two SGLang worker Pods with one GPU each. The Ray head and both workers mount a shared volume at /data."
        >
          <g className="slclu-annot">
            <text x={200} y={16} textAnchor="middle">
              kubectl apply
            </text>
            <path d="M200 22V60" />
            <polygon points="200,66 195.5,57 204.5,57" />
            <text x={536} y={16} textAnchor="middle">
              helm install, once per cluster
            </text>
            <path d="M536 22V60" />
            <polygon points="536,66 531.5,57 540.5,57" />
          </g>

          <rect className={`slclu-frame${state("frame")}`} x={8} y={34} width={784} height={372} rx={16} />
          <text className="slclu-zone-label" x={24} y={54}>
            KUBERNETES CLUSTER
          </text>

          <g className={`slclu-box slclu-box--control${state("control")}`}>
            <rect className="slclu-box__bg" x={24} y={66} width={226} height={84} rx={12} />
            <text className="slclu-zone-label" x={38} y={86}>
              CONTROL PLANE
            </text>
            <text className="slclu-box__title" x={38} y={108}>
              API server
            </text>
            {OBJECTS.reduce<{ x: number; nodes: ReactNode[] }>(
              (row, object) => {
                const width = chipWidth(object.text);
                row.nodes.push(
                  <g key={object.id} className={`slclu-object${state(object.id)}`}>
                    <rect x={row.x} y={120} width={width} height={18} rx={5} />
                    <text x={row.x + width / 2} y={132.5} textAnchor="middle">
                      {object.text}
                    </text>
                  </g>,
                );
                row.x += width + 6;
                return row;
              },
              { x: 38, nodes: [] },
            ).nodes}
          </g>

          <g className={`slclu-watch${state("watch")}`}>
            <path d="M255 108H291" />
            <polygon points="250,108 259,103.5 259,112.5" />
            <polygon points="296,108 287,103.5 287,112.5" />
            <text x={273} y={100} textAnchor="middle">
              watch
            </text>
          </g>

          <g className={`slclu-box slclu-box--addons${state("addons")}`}>
            <rect className="slclu-box__bg" x={296} y={66} width={480} height={84} rx={12} />
            <text className="slclu-zone-label" x={310} y={86}>
              CLUSTER-WIDE OPERATORS
            </text>
          </g>

          <g className={`slclu-box slclu-box--node${state("node-cpu")}`}>
            <rect className="slclu-box__bg" x={24} y={166} width={226} height={140} rx={12} />
            <text className="slclu-zone-label" x={38} y={185}>
              NODE WITHOUT GPUS
            </text>
          </g>

          <g className={`slclu-box slclu-box--node${state("node-gpu")}`}>
            <rect className="slclu-box__bg" x={266} y={166} width={510} height={140} rx={12} />
            <text className="slclu-zone-label" x={280} y={185}>
              GPU NODE
            </text>
            <text className="slclu-gpus__label" x={644} y={184} textAnchor="end">
              GPUs
            </text>
            <g className="slclu-gpus">
              <title>This node has 8 GPUs. The tutorial uses 3: one for the trainer and one for each engine.</title>
              {GPU_SLOTS.map((slot, index) => (
                <rect key={index} className={`slclu-gpu slclu-gpu--${slot}`} x={652 + index * 14} y={175} width={10} height={10} rx={2} />
              ))}
            </g>
          </g>

          <g className="slclu-mounts">
            {[357.5, 520.5, 683.5].map((x) => (
              <path key={x} d={`M${x} 264V327`} />
            ))}
            {[357.5, 520.5, 683.5].map((x) => (
              <polygon key={x} points={`${x},332 ${x - 4.5},323 ${x + 4.5},323`} />
            ))}
            <text x={366} y={322}>
              mount /data
            </text>
          </g>

          <g className={`slclu-box slclu-box--volume${state("volume")}`}>
            <rect className="slclu-box__bg" x={24} y={332} width={752} height={58} rx={12} />
            <text className="slclu-zone-label" x={38} y={352}>
              SHARED VOLUME · slime-data PVC · ReadWriteMany
            </text>
            {["models/Qwen3-0.6B/", "datasets/gsm8k/", "weight-updates/"].reduce<{ x: number; nodes: ReactNode[] }>(
              (row, file) => {
                const width = chipWidth(file);
                row.nodes.push(
                  <g key={file} className="slclu-file">
                    <rect x={row.x} y={362} width={width} height={18} rx={5} />
                    <text x={row.x + 7} y={374.5}>
                      {file}
                    </text>
                  </g>,
                );
                row.x += width + 8;
                return row;
              },
              { x: 38, nodes: [] },
            ).nodes}
          </g>

          {PODS.map((pod) => (
            <g key={pod.id} className={`slclu-pod slclu-pod--${pod.owner}${state(pod.id)}`}>
              <title>{`${pod.name} Pod. ${pod.detail}`}</title>
              <rect className="slclu-pod__halo" x={pod.x} y={pod.y} width={pod.w} height={pod.h} rx={9} />
              <rect className="slclu-pod__box" x={pod.x} y={pod.y} width={pod.w} height={pod.h} rx={9} />
              <text className="slclu-pod__name" x={pod.x + 10} y={pod.y + 15}>
                {pod.name} Pod
              </text>
              {pod.procs.map((proc, index) => {
                const y = pod.y + 22 + index * 22;
                return (
                  <g key={proc.id} className={`slclu-proc${state(proc.id)}`}>
                    <rect className="slclu-proc__lit" x={pod.x + 5} y={y - 2} width={pod.w - 10} height={20} rx={5} />
                    <g className={`slclu-chip slclu-chip--${proc.on === "GPU" ? proc.role : "cpu"}`}>
                      <rect x={pod.x + 10} y={y} width={32} height={16} rx={4} />
                      <text x={pod.x + 26} y={y + 11.5} textAnchor="middle">
                        {proc.on}
                      </text>
                    </g>
                    <text className="slclu-proc__name" x={pod.x + 49} y={y + 12}>
                      {proc.name}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>

      <div className="slclu-status" aria-live="polite">
        <p className="slclu-status__title">{active ? active.label : "Where each part runs"}</p>
        <p className="slclu-status__text">{active ? active.text : DEFAULT_TEXT}</p>
      </div>
      <div className="slclu-legend">
        <span>
          <i className="slclu-key slclu-key--dynamo" aria-hidden="true" /> Pod created by Dynamo
        </span>
        <span>
          <i className="slclu-key slclu-key--kuberay" aria-hidden="true" /> Pod created by KubeRay
        </span>
        <span>
          <i className="slclu-key slclu-key--gpu" aria-hidden="true" /> Process with its own GPU
        </span>
        <span>The scheduler can place the GPU Pods on different GPU nodes.</span>
      </div>
    </div>
  );
}

export default SlimeClusterMap;
