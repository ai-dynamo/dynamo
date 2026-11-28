/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SlimeRLLoop — the parts of the Slime integration, then the reinforcement
 * learning (RL) loop in five steps, from two boxes to Slime with Dynamo.
 *
 * The tray lists the parts. Each tile's (i) button shows a plain-language
 * summary on hover, keyboard focus, or tap, and a tile lights up on the step
 * that adds its part to the diagram.
 *
 * The steps are the loop, one round, the software, Slime on Ray, and Slime
 * with Dynamo. The first two can swap their labels for a classroom analogy,
 * and the last one flips between Slime alone and Slime with Dynamo. A box
 * that stays from one step to the next slides to its new place. Nothing
 * autoplays.
 *
 * Its numbers follow the Slime tutorial: 8 prompts x 8 samples a round, two
 * SGLang engines, and one trainer GPU. Keep them in step with the tutorial
 * pages and with SlimeIntegrationMap, which uses the same layout for the last
 * step.
 *
 * WHY "use client": it keeps the step and tooltip state. Only effects and
 * event handlers touch document, so the server renders the first step.
 *
 * The CSS lives in SlimeRLLoopStyles.tsx, where check_style_components.py
 * guards it. The page-usage example is in README.md.
 */
"use client";

import { type ReactNode, useEffect, useState } from "react";

import { SlimeRLLoopStyles } from "./SlimeRLLoopStyles";

type Tone = "green" | "amber" | "slate" | "neutral";
type PartId = "model" | "dataset" | "sglang" | "megatron" | "slime" | "ray" | "kuberay" | "dynamo";
type Point = readonly [number, number];
type Anchor = "start" | "middle" | "end";

const STEPS = [
  { label: "The loop" },
  { label: "One round" },
  { label: "The software" },
  { label: "Slime on Ray" },
  { label: "Add Dynamo" },
] as const;

const LAST = STEPS.length - 1;

/** Steps whose labels can switch to the classroom analogy. */
const ANALOGY_STEPS = 2;

const PARTS: ReadonlyArray<{
  id: PartId;
  name: string;
  kind: string;
  tone: Tone;
  /** The step that adds this part to the diagram. */
  step: number;
  about: string;
}> = [
  {
    id: "model",
    name: "Qwen3-0.6B",
    kind: "The model",
    tone: "neutral",
    step: 0,
    about:
      "A small open language model. Its 0.6 billion weights are the numbers that decide what it writes, and training changes them so that its answers get better.",
  },
  {
    id: "dataset",
    name: "GSM8K",
    kind: "Math problems",
    tone: "neutral",
    step: 1,
    about:
      "Grade-school math word problems, each with a known final answer. Because the answer is known, a few lines of code can mark the model's answer right or wrong.",
  },
  {
    id: "sglang",
    name: "SGLang",
    kind: "Inference engine",
    tone: "green",
    step: 2,
    about:
      "Runs the model on a GPU and generates text. It answers many requests at once and caches the work it did on each prompt, so a prompt that it has already seen starts faster.",
  },
  {
    id: "megatron",
    name: "Megatron-LM",
    kind: "Training engine",
    tone: "amber",
    step: 2,
    about:
      "NVIDIA's library for training models on GPUs. Given the answers and their scores, it works out how to change each weight, called the gradient, and applies the change.",
  },
  {
    id: "slime",
    name: "Slime",
    kind: "RL framework",
    tone: "amber",
    step: 2,
    about:
      "Runs the training loop. It hands problems to SGLang, grades the answers, has Megatron-LM learn from the grades, and copies the new weights back to SGLang, round after round.",
  },
  {
    id: "ray",
    name: "Ray",
    kind: "Distributed Python",
    tone: "slate",
    step: 3,
    about:
      "Runs Python across GPUs and machines. Slime starts each long-running piece, such as the trainer, as a Ray actor: a process that other code can call like an object.",
  },
  {
    id: "kuberay",
    name: "KubeRay",
    kind: "Ray on Kubernetes",
    tone: "slate",
    step: 3,
    about:
      "A Kubernetes operator for Ray. You describe a Ray cluster and the job to run on it in a RayJob. KubeRay creates the Pods and runs the job, and the tutorial has it delete the Pods when the job ends.",
  },
  {
    id: "dynamo",
    name: "Dynamo",
    kind: "Inference serving",
    tone: "green",
    step: 4,
    about:
      "Inference serving. On Kubernetes, its operator runs the SGLang engines as Pods from a DynamoGraphDeployment, and its frontend sends each request to an engine, optionally to the one that has the prompt cached.",
  },
];

/** 24 x 24 line icons, drawn with the tile's ink color. */
const ICONS: Record<PartId, ReactNode> = {
  model: (
    <>
      <circle cx="5" cy="6" r="2" />
      <circle cx="5" cy="18" r="2" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="19" cy="6" r="2" />
      <circle cx="19" cy="18" r="2" />
      <path d="M6.7 7.2 10.3 10.8M6.7 16.8l3.6-3.6M13.7 10.8l3.6-3.6M13.7 13.2l3.6 3.6" />
    </>
  ),
  dataset: (
    <>
      <path d="M6 3h9l4 4v14H6z" />
      <path d="M15 3v4h4M9.5 11.5h5M12 9v5M9.5 17.5h5" />
    </>
  ),
  sglang: (
    <>
      <path d="M4 5h16v11h-9l-5 4v-4H4z" />
      <path d="M8 9h8M8 12h5" />
    </>
  ),
  megatron: <path d="M7 7v10M4 9.5v5M17 7v10M20 9.5v5M7 12h10" />,
  slime: (
    <>
      <path d="M19.5 12a7.5 7.5 0 0 1-13 5.1M4.5 12a7.5 7.5 0 0 1 13-5.1" />
      <path d="M17.5 3.5v3.6h-3.6M6.5 20.5v-3.6h3.6" />
    </>
  ),
  ray: (
    <>
      <circle cx="12" cy="12" r="2.6" />
      <circle cx="4.5" cy="5" r="1.7" />
      <circle cx="19.5" cy="5" r="1.7" />
      <circle cx="4.5" cy="19" r="1.7" />
      <circle cx="19.5" cy="19" r="1.7" />
      <path d="M5.8 6.2 10 10.2M18.2 6.2 14 10.2M5.8 17.8 10 13.8M18.2 17.8 14 13.8" />
    </>
  ),
  kuberay: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 3.5V6M12 18v2.5M3.5 12H6M18 12h2.5M6 6l1.8 1.8M16.2 16.2 18 18M6 18l1.8-1.8M16.2 7.8 18 6" />
    </>
  ),
  dynamo: (
    <>
      <path d="M3 12h5M8 12c3 0 3-6 7-6h4M8 12h11M8 12c3 0 3 6 7 6h4" />
      <path d="M17 4l2 2-2 2M17 10l2 2-2 2M17 16l2 2-2 2" />
    </>
  ),
};

type NodeSpec = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  tone: Tone;
  title: string;
  sub: readonly string[];
  /** Step number badge, for the round's stations. */
  num?: number;
  /** What the part computes on. */
  chip?: "CPU" | "GPU";
  /** Holds its own copy of the model weights. */
  weights?: boolean;
};

type Label = {
  text: string;
  x: number;
  y: number;
  anchor?: Anchor;
  /** The zone tint under the label, so its halo matches. Default: the panel. */
  on?: "slate" | "green";
};

type WireSpec = {
  id: string;
  tone: Tone;
  d: string;
  heads: readonly string[];
  dashed?: boolean;
  label?: Label;
  num?: { n: number; x: number; y: number };
};

type ZoneSpec = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  tone: Tone;
  label: string;
  owner?: string;
  /** A dashed group inside another zone. */
  inner?: boolean;
  chips?: readonly string[];
};

type Scene = {
  aria: string;
  zones: readonly ZoneSpec[];
  nodes: readonly NodeSpec[];
  wires: readonly WireSpec[];
  center?: Label;
};

const r1 = (n: number) => Math.round(n * 10) / 10;
const pt = ([x, y]: Point) => `${r1(x)} ${r1(y)}`;

/** An arrowhead whose tip is `tip`, pointing away from `from`. */
function arrowHead(tip: Point, from: Point): string {
  const [x2, y2] = tip;
  const length = Math.hypot(x2 - from[0], y2 - from[1]) || 1;
  const ux = (x2 - from[0]) / length;
  const uy = (y2 - from[1]) / length;
  const size = 9;
  const bx = x2 - ux * size;
  const by = y2 - uy * size;
  return [
    [x2, y2],
    [bx - uy * size * 0.5, by + ux * size * 0.5],
    [bx + uy * size * 0.5, by - ux * size * 0.5],
  ]
    .map(([x, y]) => `${r1(x)},${r1(y)}`)
    .join(" ");
}

/** `tip`, moved 5 units back toward `from`, so the line ends inside its arrowhead. */
function shorten(tip: Point, from: Point): Point {
  const length = Math.hypot(tip[0] - from[0], tip[1] - from[1]) || 1;
  return [tip[0] - ((tip[0] - from[0]) / length) * 5, tip[1] - ((tip[1] - from[1]) / length) * 5];
}

/** A polyline with an arrowhead at its end, and one at its start too when `both`. */
function line(points: readonly Point[], both = false) {
  const n = points.length;
  const start = both ? shorten(points[0], points[1]) : points[0];
  const end = shorten(points[n - 1], points[n - 2]);
  const d = [start, ...points.slice(1, -1), end].map((p, i) => `${i ? "L" : "M"}${pt(p)}`).join(" ");
  const heads = [arrowHead(points[n - 1], points[n - 2])];
  if (both) heads.push(arrowHead(points[0], points[1]));
  return { d, heads };
}

/** A cubic curve from p0 to p1 with an arrowhead at p1. */
function curve(p0: Point, c1: Point, c2: Point, p1: Point) {
  const end = shorten(p1, c2);
  const c2End: Point = [c2[0] + end[0] - p1[0], c2[1] + end[1] - p1[1]];
  return { d: `M${pt(p0)} C${pt(c1)} ${pt(c2End)} ${pt(end)}`, heads: [arrowHead(p1, c2)] };
}

// The software layout, shared by the last three steps. SlimeIntegrationMap
// draws the tutorial with the same arrangement.
const SLIME_BOX = { x: 26, y: 62, w: 200, h: 84 };
const TRAINER_BOX = { x: 26, y: 234, w: 200, h: 84 };
const ROUTER_BOX = { x: 338, y: 66, w: 152, h: 78 };
const ENGINE_BOXES = [
  { x: 562, y: 84, w: 204, h: 78 },
  { x: 562, y: 200, w: 204, h: 78 },
] as const;
const ENGINE_ZONE: ZoneSpec = { id: "engines", x: 548, y: 52, w: 234, h: 248, tone: "green", label: "SGLANG ENGINES", inner: true };

function engines(sub: string): NodeSpec[] {
  return ENGINE_BOXES.map((box, index): NodeSpec => ({
    id: `engine${index}`,
    ...box,
    tone: "green",
    title: `SGLang engine ${index}`,
    sub: [sub],
    chip: "GPU",
    weights: true,
  }));
}

function batch(on?: "slate"): WireSpec {
  return {
    id: "batch",
    tone: "amber",
    ...line([
      [126, 146],
      [126, 234],
    ]),
    label: { text: "answers + scores", x: 136, y: 194, anchor: "start", on },
  };
}

const ROUTES: readonly WireSpec[] = [
  {
    id: "route0",
    tone: "green",
    ...line(
      [
        [490, 96],
        [562, 118],
      ],
      true,
    ),
  },
  {
    id: "route1",
    tone: "green",
    ...line(
      [
        [490, 120],
        [522, 120],
        [522, 236],
        [562, 236],
      ],
      true,
    ),
  },
];

function loopScene(analogy: boolean): Scene {
  return {
    aria: analogy
      ? "Two boxes in a loop: the kid takes a quiz and passes the answers to the parents, and the parents help the kid practice."
      : "Two boxes in a loop: the model sends its answers to the trainer, and the trainer sends new weights back to the model.",
    zones: [],
    nodes: [
      {
        id: "trainer",
        x: 70,
        y: 158,
        w: 214,
        h: 92,
        tone: "amber",
        title: analogy ? "Parents" : "Trainer",
        sub: [analogy ? "help the kid practice" : "improves the model"],
      },
      {
        id: "engine0",
        x: 516,
        y: 158,
        w: 214,
        h: 92,
        tone: "green",
        title: analogy ? "Kid" : "Model",
        sub: [analogy ? "takes a quiz" : "answers questions"],
      },
    ],
    wires: [
      {
        id: "weights",
        tone: "amber",
        ...curve([177, 158], [177, 62], [623, 62], [623, 158]),
        label: { text: analogy ? "what the kid practiced" : "new weights", x: 400, y: 76 },
      },
      {
        id: "answers",
        tone: "green",
        ...curve([623, 250], [623, 346], [177, 346], [177, 250]),
        label: { text: analogy ? "quiz answers" : "answers", x: 400, y: 344 },
      },
    ],
    center: { text: "↻ repeat", x: 400, y: 209 },
  };
}

function roundScene(analogy: boolean): Scene {
  return {
    aria: analogy
      ? "One round: the quiz's 8 questions go to the kid, the kid's 64 answers go to the teacher, the marked answers go to the parents, and what the kid practiced goes back to the kid."
      : "One round: GSM8K supplies 8 problems to the model, the model's 64 answers go to a grader, the scored answers go to the trainer, and the trainer sends new weights to the model.",
    zones: [],
    nodes: [
      {
        id: "dataset",
        x: 545,
        y: 22,
        w: 184,
        h: 56,
        tone: "neutral",
        num: 1,
        title: analogy ? "Quiz" : "GSM8K",
        sub: [analogy ? "8 questions" : "8 problems a round"],
      },
      {
        id: "engine0",
        x: 530,
        y: 152,
        w: 214,
        h: 92,
        tone: "green",
        num: 2,
        title: analogy ? "Kid" : "Model",
        sub: [analogy ? "answers each question 8 ways" : "answers each problem 8 times"],
      },
      {
        id: "slime",
        x: 40,
        y: 50,
        w: 214,
        h: 84,
        tone: "amber",
        num: 3,
        title: analogy ? "Teacher" : "Grader",
        sub: [analogy ? "marks each answer ✓ or ✗" : "1 if right, 0 if wrong"],
      },
      {
        id: "trainer",
        x: 40,
        y: 254,
        w: 214,
        h: 84,
        tone: "amber",
        num: 4,
        title: analogy ? "Parents" : "Trainer",
        sub: [analogy ? "practice what worked" : "learns from the scores"],
      },
    ],
    wires: [
      {
        id: "problems",
        tone: "neutral",
        ...line([
          [637, 78],
          [637, 152],
        ]),
        label: { text: analogy ? "8 questions" : "8 problems", x: 647, y: 120, anchor: "start" },
      },
      {
        id: "answers",
        tone: "green",
        ...line([
          [530, 176],
          [400, 176],
          [400, 92],
          [254, 92],
        ]),
        label: { text: "64 answers", x: 327, y: 84 },
      },
      {
        id: "scores",
        tone: "amber",
        ...line([
          [147, 134],
          [147, 254],
        ]),
        label: { text: analogy ? "marked answers" : "answers + scores", x: 157, y: 198, anchor: "start" },
      },
      {
        id: "weights",
        tone: "amber",
        ...line([
          [254, 296],
          [637, 296],
          [637, 244],
        ]),
        label: { text: analogy ? "what the kid practiced" : "new weights", x: 446, y: 288 },
        num: { n: 5, x: 300, y: 296 },
      },
    ],
  };
}

function softwareScene(): Scene {
  return {
    aria: "Slime sends problems to two SGLang engines and gets the answers back, grades them, and passes them to Megatron-LM, which sends new weights to the engines.",
    zones: [ENGINE_ZONE],
    nodes: [
      { id: "slime", ...SLIME_BOX, tone: "amber", title: "Slime", sub: ["runs the loop", "grades each answer"], chip: "CPU" },
      {
        id: "trainer",
        ...TRAINER_BOX,
        tone: "amber",
        title: "Megatron-LM",
        sub: ["computes the update"],
        chip: "GPU",
        weights: true,
      },
      ...engines("generates answers"),
    ],
    wires: [
      {
        id: "requests",
        tone: "green",
        ...line(
          [
            [226, 100],
            [548, 100],
          ],
          true,
        ),
        label: { text: "problems out, answers back", x: 387, y: 92 },
      },
      batch(),
      {
        id: "weights",
        tone: "amber",
        ...line([
          [226, 276],
          [548, 276],
        ]),
        label: { text: "new weights, copied", x: 387, y: 268 },
      },
    ],
  };
}

function rayScene(): Scene {
  return {
    aria: "Inside one Ray cluster, Slime sends requests through its router to two SGLang engines, and Megatron-LM broadcasts new weights to the engines over NCCL.",
    zones: [
      {
        id: "ray",
        x: 10,
        y: 16,
        w: 780,
        h: 314,
        tone: "slate",
        label: "RAY CLUSTER",
        owner: "KubeRay creates it from a RayJob",
      },
      ENGINE_ZONE,
    ],
    nodes: [
      {
        id: "slime",
        ...SLIME_BOX,
        tone: "amber",
        title: "Slime",
        sub: ["Ray driver · runs the loop", "grades each answer"],
        chip: "CPU",
      },
      { id: "router", ...ROUTER_BOX, tone: "green", title: "Router", sub: ["Slime's SGLang router"], chip: "CPU" },
      {
        id: "trainer",
        ...TRAINER_BOX,
        tone: "amber",
        title: "Megatron-LM",
        sub: ["Ray actor · trains"],
        chip: "GPU",
        weights: true,
      },
      ...engines("Ray actor · generates"),
    ],
    wires: [
      {
        id: "requests",
        tone: "green",
        ...line(
          [
            [226, 100],
            [338, 100],
          ],
          true,
        ),
        label: { text: "requests", x: 282, y: 92, on: "slate" },
      },
      ...ROUTES,
      batch("slate"),
      {
        id: "weights",
        tone: "amber",
        ...line([
          [226, 276],
          [548, 276],
        ]),
        label: { text: "new weights · NCCL, GPU to GPU", x: 387, y: 268, on: "slate" },
      },
    ],
  };
}

function dynamoScene(): Scene {
  return {
    aria: "The Ray head Pod holds Slime and Megatron-LM. A DynamoGraphDeployment holds the Dynamo frontend and two SGLang engines. Slime sends rollout requests to the frontend, pauses and reloads each engine directly, and passes new weights through a shared volume.",
    zones: [
      { id: "ray", x: 10, y: 16, w: 236, h: 314, tone: "slate", label: "RAY HEAD POD", owner: "KubeRay" },
      {
        id: "dgd",
        x: 322,
        y: 16,
        w: 468,
        h: 314,
        tone: "green",
        label: "DYNAMOGRAPHDEPLOYMENT",
        owner: "Dynamo operator",
      },
      ENGINE_ZONE,
      {
        id: "volume",
        x: 10,
        y: 350,
        w: 780,
        h: 44,
        tone: "neutral",
        label: "SHARED VOLUME · /data",
        owner: "slime-data PVC",
        chips: ["weight-updates/"],
      },
    ],
    nodes: [
      {
        id: "slime",
        ...SLIME_BOX,
        tone: "amber",
        title: "Slime",
        sub: ["runs the loop · grades", "pauses, reloads engines"],
        chip: "CPU",
      },
      { id: "router", ...ROUTER_BOX, tone: "green", title: "Dynamo frontend", sub: ["routes requests"], chip: "CPU" },
      {
        id: "trainer",
        ...TRAINER_BOX,
        tone: "amber",
        title: "Megatron-LM",
        sub: ["trains · writes weights"],
        chip: "GPU",
        weights: true,
      },
      ...engines("Pod · Dynamo sidecar"),
    ],
    wires: [
      {
        id: "requests",
        tone: "green",
        ...line(
          [
            [226, 100],
            [338, 100],
          ],
          true,
        ),
        label: { text: "/generate", x: 284, y: 92 },
      },
      ...ROUTES,
      batch("slate"),
      {
        id: "control",
        tone: "amber",
        dashed: true,
        ...line([
          [226, 132],
          [284, 132],
          [284, 288],
          [548, 288],
        ]),
        label: { text: "pause · reload · resume", x: 416, y: 280, on: "green" },
      },
      {
        id: "write",
        tone: "amber",
        ...line([
          [126, 318],
          [126, 350],
        ]),
        label: { text: "write", x: 136, y: 340, anchor: "start" },
      },
      {
        id: "load",
        tone: "green",
        ...line([
          [665, 350],
          [665, 300],
        ]),
        label: { text: "load", x: 675, y: 330, anchor: "start" },
      },
    ],
  };
}

function buildScene(index: number, analogy: boolean): Scene {
  switch (index) {
    case 0:
      return loopScene(analogy);
    case 1:
      return roundScene(analogy);
    case 2:
      return softwareScene();
    case 3:
      return rayScene();
    default:
      return dynamoScene();
  }
}

const COMPARE: ReadonlyArray<readonly [string, string, string]> = [
  ["SGLang engines", "Ray actors that Slime starts", "Pods that the Dynamo operator runs"],
  ["Rollout routing", "Slime's SGLang router", "Dynamo frontend, round-robin or KV-aware"],
  ["Weight sync", "NCCL broadcast, GPU to GPU", "Files on a shared volume, in the tutorial"],
  ["After the job", "Deleted with the Ray cluster", "Keep serving until you delete the DGD"],
  ["Loop, grading, training", "Slime and Megatron-LM", "Unchanged"],
];

function stepText(step: number, analogy: boolean, dynamo: boolean): ReactNode {
  switch (step) {
    case 0:
      return analogy ? (
        <p>
          Think of a kid getting better at math. The kid takes a quiz, the parents help the kid practice what the quiz
          showed, and then the kid takes the next quiz. The kid is the model, and the parents are the trainer.
        </p>
      ) : (
        <p>
          Reinforcement learning (RL) trains a model by practice. The model answers questions, the trainer adjusts the
          model&apos;s weights according to how good the answers were, and the adjusted model answers the next
          questions. Generating the answers is called a <b>rollout</b>.
        </p>
      );
    case 1:
      return analogy ? (
        <>
          <p>
            ① The quiz has 8 questions. ② The kid answers each one 8 different ways. ③ The teacher marks every answer
            right or wrong, and the marked quiz goes home. ④ The parents compare the right and wrong answers to each
            question and practice what the right ones did. ⑤ The kid takes the next quiz.
          </p>
          <p className="slrl-caveat">
            <b>Where the analogy breaks:</b> the teacher only marks; working out what to change is the parents&apos;
            job, and that&apos;s where the trainer does its math. And the model exists twice: one copy takes the
            quizzes and another studies, so after each lesson the studied weights are copied to the quiz-taker.
          </p>
        </>
      ) : (
        <p>
          ① GSM8K supplies 8 problems. ② The model answers each one 8 times, 64 answers in all. ③ A grader scores each
          answer: 1 if the final number is right, 0 if not. ④ The trainer compares the 8 answers to each problem and
          adjusts the weights so that the above-average answers become more likely and the rest less likely. That
          comparison within a group is GRPO, the tutorial&apos;s algorithm. ⑤ The new weights go back to the model for
          the next round.
        </p>
      );
    case 2:
      return (
        <p>
          Slime runs the loop. It sends the problems to SGLang, grades the answers with a rule-based math check, and
          passes the graded answers to Megatron-LM, which computes the weight update on its GPU. SGLang and Megatron-LM
          each hold their own copy of the weights, so after every update Slime copies the new weights to the engines.
          That copy is the <b>weight sync</b>.
        </p>
      );
    case 3:
      return (
        <p>
          On its own, Slime runs everything in one Ray cluster. Its driver script starts the Megatron-LM trainer and
          each SGLang engine as a Ray actor on its own GPUs, plus a router that picks an engine for each request.
          After each update, the trainer broadcasts the new weights to the engines&apos; GPUs over NCCL. On Kubernetes,
          KubeRay creates the Ray cluster from a RayJob. Slime can also run training and inference on the same GPUs, in
          turns, with <code>--colocate</code>; the tutorial keeps them apart.
        </p>
      );
    default:
      return dynamo ? (
        <p>
          With Dynamo, the SGLang engines move out of Slime&apos;s Ray cluster. The Dynamo operator runs each engine as
          a Pod from a DynamoGraphDeployment, and the Dynamo frontend sends each rollout request to an engine, either
          in turn or, with KV-aware routing, by what each engine already has cached. Slime keeps the loop, the
          grading, and Megatron-LM, and calls each engine directly to pause it, load new weights, and resume it.
        </p>
      ) : (
        <p>
          Before Dynamo, Slime starts the SGLang engines and their router inside its own Ray cluster and broadcasts the
          weights to them over NCCL. Switch to <b>With Dynamo</b> to see what moves.
        </p>
      );
  }
}

/**
 * Text rows inside a node. A node with a chip or a weights tag keeps its text
 * at the top, clear of the bottom row; any other node centers its text.
 */
function nodeText(node: NodeSpec) {
  const lines = node.sub.length;
  const top = node.chip || node.weights ? 10 : Math.round((node.h - 18 - 16 * lines) / 2);
  return {
    titleY: top + 14,
    subY: (index: number) => top + 31 + 16 * index,
    weightsY: top + 18 + 16 * lines + 4,
  };
}

function Diagram({ scene }: { scene: Scene }) {
  return (
    <svg className="slrl-svg" viewBox="0 0 800 410" role="img" aria-label={scene.aria}>
      {scene.zones.map((zone) => (
        <g key={zone.id} className={`slrl-zone slrl-tone--${zone.tone}${zone.inner ? " slrl-zone--inner" : ""}`}>
          <rect x={zone.x} y={zone.y} width={zone.w} height={zone.h} rx={zone.inner ? 12 : 14} />
          <text className="slrl-zone__label" x={zone.x + 16} y={zone.y + (zone.h < 60 ? 27 : 20)}>
            {zone.label}
          </text>
          {zone.owner ? (
            <text
              className="slrl-zone__owner"
              x={zone.x + zone.w - 14}
              y={zone.y + (zone.h < 60 ? 27 : 20)}
              textAnchor="end"
            >
              {zone.owner}
            </text>
          ) : null}
          {(zone.chips ?? []).map((chip) => (
            <g key={chip} className="slrl-file">
              <rect x={zone.x + 196} y={zone.y + 13} width={chip.length * 6.3 + 16} height={18} rx={5} />
              <text x={zone.x + 204} y={zone.y + 25.5}>
                {chip}
              </text>
            </g>
          ))}
        </g>
      ))}

      {scene.wires.map((wire) => (
        <g key={wire.id + wire.d} className={`slrl-wire slrl-tone--${wire.tone}${wire.dashed ? " slrl-wire--dashed" : ""}`}>
          <path d={wire.d} />
          {wire.heads.map((head) => (
            <polygon key={head} points={head} />
          ))}
        </g>
      ))}

      {scene.nodes.map((node) => {
        const text = nodeText(node);
        return (
          <g
            key={node.id}
            className={`slrl-node slrl-tone--${node.tone}`}
            style={{ transform: `translate(${node.x}px, ${node.y}px)` }}
          >
            <rect className="slrl-node__box" width={node.w} height={node.h} rx={10} />
            <text className="slrl-node__title" x={14} y={text.titleY}>
              {node.title}
            </text>
            {node.sub.map((line, index) => (
              <text key={line} className="slrl-node__sub" x={14} y={text.subY(index)}>
                {line}
              </text>
            ))}
            {node.weights ? (
              <g className="slrl-weights">
                <rect x={14} y={text.weightsY} width={58} height={16} rx={8} />
                <text x={43} y={text.weightsY + 11.5} textAnchor="middle">
                  weights
                </text>
              </g>
            ) : null}
            {node.chip ? (
              <g className={`slrl-chip slrl-chip--${node.chip.toLowerCase()}`}>
                <rect x={node.w - 46} y={node.h - 27} width={34} height={17} rx={4} />
                <text x={node.w - 29} y={node.h - 15} textAnchor="middle">
                  {node.chip}
                </text>
              </g>
            ) : null}
            {node.num ? (
              <g className="slrl-num">
                <circle cx={2} cy={2} r={11} />
                <text x={2} y={6.5} textAnchor="middle">
                  {node.num}
                </text>
              </g>
            ) : null}
          </g>
        );
      })}

      {scene.wires.map((wire) => (
        <g key={`label-${wire.id}-${wire.d}`} className={`slrl-tone--${wire.tone}`}>
          {wire.label ? (
            <text
              className={`slrl-wire__label${wire.label.on ? ` slrl-wire__label--on-${wire.label.on}` : ""}`}
              x={wire.label.x}
              y={wire.label.y}
              textAnchor={wire.label.anchor ?? "middle"}
            >
              {wire.label.text}
            </text>
          ) : null}
          {wire.num ? (
            <g className="slrl-num">
              <circle cx={wire.num.x} cy={wire.num.y} r={11} />
              <text x={wire.num.x} y={wire.num.y + 4.5} textAnchor="middle">
                {wire.num.n}
              </text>
            </g>
          ) : null}
        </g>
      ))}

      {scene.center ? (
        <text className="slrl-center" x={scene.center.x} y={scene.center.y} textAnchor="middle">
          {scene.center.text}
        </text>
      ) : null}
    </svg>
  );
}

export function SlimeRLLoop() {
  const [step, setStep] = useState(0);
  const [analogy, setAnalogy] = useState(false);
  const [dynamo, setDynamo] = useState(true);
  // The part whose summary was opened by a click or tap. Hover and keyboard
  // focus show a summary through CSS alone.
  const [tip, setTip] = useState<PartId | null>(null);

  useEffect(() => {
    if (!tip) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setTip(null);
    };
    const onPointer = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Element && target.closest(`[data-part="${tip}"]`)) return;
      setTip(null);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
    };
  }, [tip]);

  const showAnalogy = analogy && step < ANALOGY_STEPS;
  const scene = buildScene(step === LAST && !dynamo ? LAST - 1 : step, showAnalogy);
  const next = () => setStep(step === LAST ? 0 : step + 1);

  return (
    <div className="slrl not-prose">
      <SlimeRLLoopStyles />

      <p className="slrl-eyebrow">The parts</p>
      <ul className="slrl-tray">
        {PARTS.map((part) => {
          const state = part.step === step ? "is-new" : part.step < step ? "is-seen" : "is-later";
          return (
            <li
              key={part.id}
              data-part={part.id}
              className={`slrl-part slrl-tone--${part.tone} ${state}${tip === part.id ? " is-open" : ""}`}
            >
              <span className="slrl-part__icon" aria-hidden="true">
                <svg viewBox="0 0 24 24">{ICONS[part.id]}</svg>
              </span>
              <span className="slrl-part__text">
                <span className="slrl-part__name">{part.name}</span>
                <span className="slrl-part__kind">{part.kind}</span>
              </span>
              <button
                type="button"
                className="slrl-part__info"
                aria-label={`What is ${part.name}?`}
                aria-describedby={`slrl-tip-${part.id}`}
                aria-expanded={tip === part.id}
                onClick={() => setTip(tip === part.id ? null : part.id)}
              >
                i
              </button>
              <span role="tooltip" id={`slrl-tip-${part.id}`} className="slrl-tip">
                {part.about}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="slrl-caption">
        Hover or tap <span className="slrl-caption__i">i</span> for a plain-language summary. Each part lights up on
        the step that adds it.
      </p>

      <section className="slrl-world" aria-label="The RL loop, step by step">
        <div className="slrl-head">
          <div className="slrl-head__title">
            <p className="slrl-eyebrow">
              Step {step + 1} of {STEPS.length}
            </p>
            <p className="slrl-title">{STEPS[step].label}</p>
          </div>
          <div className="slrl-controls">
            {step < ANALOGY_STEPS ? (
              <button
                type="button"
                className="slrl-toggle"
                aria-pressed={analogy}
                onClick={() => setAnalogy(!analogy)}
              >
                <span className="slrl-toggle__box" aria-hidden="true">
                  {analogy ? "✓" : ""}
                </span>
                Classroom analogy
              </button>
            ) : null}
            {step === LAST ? (
              <div className="slrl-seg" role="group" aria-label="Compare">
                <button type="button" aria-pressed={!dynamo} onClick={() => setDynamo(false)}>
                  Slime alone
                </button>
                <button type="button" aria-pressed={dynamo} onClick={() => setDynamo(true)}>
                  With Dynamo
                </button>
              </div>
            ) : null}
            <button type="button" className="slrl-nav" onClick={() => setStep(step - 1)} disabled={step === 0}>
              <span aria-hidden="true">←</span> Back
            </button>
            <button type="button" className="slrl-nav slrl-nav--next" onClick={next}>
              {step === LAST ? "Start over" : `Next: ${STEPS[step + 1].label}`}{" "}
              <span aria-hidden="true">{step === LAST ? "↺" : "→"}</span>
            </button>
          </div>
        </div>

        <ol className="slrl-steps" aria-label="Steps">
          {STEPS.map((item, index) => (
            <li key={item.label}>
              <button
                type="button"
                className={`slrl-step${index < step ? " is-done" : ""}`}
                aria-current={index === step ? "step" : undefined}
                onClick={() => setStep(index)}
              >
                <span className="slrl-step__num" aria-hidden="true">
                  {index + 1}
                </span>
                {item.label}
              </button>
            </li>
          ))}
        </ol>

        <div className="slrl-stage">
          <Diagram scene={scene} />
        </div>

        <div className="slrl-status" aria-live="polite">
          {stepText(step, showAnalogy, dynamo)}
          {step === LAST ? (
            <table className="slrl-compare">
              <thead>
                <tr>
                  <th scope="col">
                    <span className="slrl-sr">Part</span>
                  </th>
                  <th scope="col" className={dynamo ? "" : "is-current"}>
                    Slime alone
                  </th>
                  <th scope="col" className={dynamo ? "is-current" : ""}>
                    With Dynamo
                  </th>
                </tr>
              </thead>
              <tbody>
                {COMPARE.map(([part, alone, withDynamo]) => (
                  <tr key={part}>
                    <th scope="row">{part}</th>
                    <td className={dynamo ? "" : "is-current"}>{alone}</td>
                    <td className={dynamo ? "is-current" : ""}>{withDynamo}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      </section>
    </div>
  );
}

export default SlimeRLLoop;
