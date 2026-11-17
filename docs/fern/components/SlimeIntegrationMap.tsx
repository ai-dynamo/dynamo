/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SlimeIntegrationMap — who owns each part of the Slime integration, and one
 * GRPO round of the Slime tutorial played step by step.
 *
 * The top row has one card per owner: Dynamo serves the rollouts, Slime
 * trains, and KubeRay runs the training job. Hovering a card highlights what
 * that owner runs in the diagram below it.
 *
 * The diagram plays one round on demand: generate, score, train, write
 * weights, reload engines. It never autoplays, stops after each round,
 * pauses when the tab is hidden, and draws no moving packets for readers who
 * prefer reduced motion. Its numbers follow the Slime tutorial pages: 10
 * rounds of 8 prompts x 8 samples, two engines, and one weight version per
 * sync, starting at v1. Keep them in step with the tutorial.
 *
 * WHY "use client": it keeps playback state and runs a timer. Only effects
 * and event handlers touch window or document, so the server renders the
 * idle round.
 *
 * The CSS lives in SlimeIntegrationMapStyles.tsx, where
 * check_style_components.py guards it. The page-usage example is in
 * README.md.
 */
"use client";

import { type ReactNode, useEffect, useState } from "react";

import { SlimeIntegrationMapStyles } from "./SlimeIntegrationMapStyles";

type Owner = "dynamo" | "slime" | "kuberay";
type NodeId = "slime" | "trainer" | "frontend" | "engine0" | "engine1";
type WireId = "generate" | "route0" | "route1" | "batch" | "control" | "write" | "load";
type StepId = "generate" | "score" | "train" | "write" | "reload";
type Tone = "green" | "amber";
type Point = readonly [number, number];

/** The tutorial runs 10 rounds (--num-rollout 10). */
const ROUNDS = 10;
/** Time on each step while playing. The slmap-progress animation uses the same 5s. */
const STEP_MS = 5000;
/** Seconds for a packet to travel one wire. */
const TRAVEL_S = 1.8;

const OWNER_NAMES: Record<Owner, string> = {
  dynamo: "Dynamo",
  slime: "Slime",
  kuberay: "KubeRay",
};

const OWNERS: ReadonlyArray<{
  id: Owner;
  num: string;
  role: string;
  text: string;
  flow: readonly [string, string, string];
}> = [
  {
    id: "dynamo",
    num: "01",
    role: "Serves the rollouts",
    text: "The Dynamo operator runs the frontend and the SGLang engine Pods from a DynamoGraphDeployment. The frontend routes each rollout request to an engine.",
    flow: ["Frontend", "⇉", "SGLang engines"],
  },
  {
    id: "slime",
    num: "02",
    role: "Trains the policy",
    text: "Slime runs the GRPO loop. It sends the rollouts, scores them, trains with Megatron-LM, and reloads each engine with the new weights.",
    flow: ["Megatron-LM", "⇄", "SGLang"],
  },
  {
    id: "kuberay",
    num: "03",
    role: "Runs the training job",
    text: "KubeRay creates a Ray cluster for the RayJob, runs Slime's training script on its head Pod, and deletes the cluster when the job ends.",
    flow: ["RayJob", "→", "Ray head Pod"],
  },
];

const NODES: ReadonlyArray<{
  id: NodeId;
  owner: Owner;
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub: readonly string[];
}> = [
  { id: "slime", owner: "slime", x: 30, y: 80, w: 214, h: 76, title: "Slime", sub: ["rollouts · rewards · weight sync"] },
  { id: "trainer", owner: "slime", x: 30, y: 200, w: 214, h: 76, title: "Megatron-LM", sub: ["Slime's trainer · 1 GPU"] },
  { id: "frontend", owner: "dynamo", x: 362, y: 80, w: 150, h: 76, title: "Frontend", sub: ["Dynamo · port 8000", "round-robin"] },
  { id: "engine0", owner: "dynamo", x: 566, y: 98, w: 194, h: 72, title: "Engine 0", sub: ["SGLang + Dynamo sidecar"] },
  { id: "engine1", owner: "dynamo", x: 566, y: 194, w: 194, h: 72, title: "Engine 1", sub: ["SGLang + Dynamo sidecar"] },
];

type WireLabel = {
  text: string;
  x: number;
  y: number;
  anchor: "start" | "middle";
  /** What the label sits on, so its halo matches the background. */
  on: "panel" | "dgd" | "ray";
};

const WIRE_SPECS: ReadonlyArray<{
  id: WireId;
  points: readonly Point[];
  tone: Tone;
  dashed?: boolean;
  label?: WireLabel;
}> = [
  {
    id: "generate",
    points: [[244, 112], [362, 112]],
    tone: "green",
    label: { text: "/generate", x: 304, y: 103, anchor: "middle", on: "panel" },
  },
  { id: "route0", points: [[512, 110], [566, 132]], tone: "green" },
  { id: "route1", points: [[512, 126], [566, 226]], tone: "green" },
  {
    id: "batch",
    points: [[137, 156], [137, 200]],
    tone: "amber",
    label: { text: "samples + rewards", x: 147, y: 182, anchor: "start", on: "ray" },
  },
  {
    id: "control",
    points: [[244, 142], [304, 142], [304, 275], [552, 275]],
    tone: "amber",
    dashed: true,
    label: { text: "pause · load · resume · :30000", x: 446, y: 266, anchor: "middle", on: "dgd" },
  },
  {
    id: "write",
    points: [[137, 276], [137, 326]],
    tone: "amber",
    label: { text: "write", x: 147, y: 315, anchor: "start", on: "panel" },
  },
  {
    id: "load",
    points: [[663, 326], [663, 284]],
    tone: "green",
    label: { text: "load", x: 673, y: 312, anchor: "start", on: "panel" },
  },
];

const round1 = (n: number) => Math.round(n * 10) / 10;

/** A path that stops short of its last point, plus an arrowhead whose tip is that point. */
function wireGeometry(points: readonly Point[]) {
  const [x2, y2] = points[points.length - 1];
  const [x1, y1] = points[points.length - 2];
  const length = Math.hypot(x2 - x1, y2 - y1);
  const ux = (x2 - x1) / length;
  const uy = (y2 - y1) / length;
  const trimmed = [...points.slice(0, -1), [x2 - ux * 5, y2 - uy * 5] as const];
  const d = trimmed.map(([x, y], i) => `${i ? "L" : "M"}${round1(x)} ${round1(y)}`).join(" ");
  const size = 9;
  const bx = x2 - ux * size;
  const by = y2 - uy * size;
  const px = -uy * size * 0.5;
  const py = ux * size * 0.5;
  const head = [
    [x2, y2],
    [bx + px, by + py],
    [bx - px, by - py],
  ]
    .map(([x, y]) => `${round1(x)},${round1(y)}`)
    .join(" ");
  return { d, head };
}

const WIRES = WIRE_SPECS.map((spec) => ({ ...spec, ...wireGeometry(spec.points) }));
const WIRE_PATHS = Object.fromEntries(WIRES.map((wire) => [wire.id, wire.d])) as Record<WireId, string>;

const checkpoint = (version: number) => `weight_v${String(version).padStart(6, "0")}`;

const STEPS: ReadonlyArray<{
  id: StepId;
  label: string;
  owners: readonly Owner[];
  nodes: readonly NodeId[];
  wires: readonly WireId[];
  volume?: boolean;
  text: (round: number) => ReactNode;
}> = [
  {
    id: "generate",
    label: "Generate",
    owners: ["slime", "dynamo"],
    nodes: ["slime", "frontend", "engine0", "engine1"],
    wires: ["generate", "route0", "route1"],
    text: (round) => (
      <>
        Slime sends 64 requests to the Dynamo frontend, 8 samples for each of 8 GSM8K prompts. The frontend
        alternates them between the two engines, which generate with weights v{round} and stream back tokens and
        log-probabilities.
      </>
    ),
  },
  {
    id: "score",
    label: "Score",
    owners: ["slime"],
    nodes: ["slime"],
    wires: [],
    text: () => (
      <>
        Slime scores each answer with a rule-based math reward. GRPO compares the 8 rewards for each prompt, and
        answers that beat their prompt's average get a positive advantage.
      </>
    ),
  },
  {
    id: "train",
    label: "Train",
    owners: ["slime"],
    nodes: ["slime", "trainer"],
    wires: ["batch"],
    text: () => <>Megatron-LM trains one GRPO step on the 64 samples, on the Ray head Pod's GPU.</>,
  },
  {
    id: "write",
    label: "Write weights",
    owners: ["slime"],
    nodes: ["trainer"],
    wires: ["write"],
    volume: true,
    text: (round) => (
      <>
        Slime saves the new weights to the shared volume as a full Hugging Face checkpoint,{" "}
        <code>{checkpoint(round + 1)}</code>, about 1.2 GB for Qwen3-0.6B.
      </>
    ),
  },
  {
    id: "reload",
    label: "Reload engines",
    owners: ["slime"],
    nodes: ["slime", "engine0", "engine1"],
    wires: ["control", "load"],
    volume: true,
    text: (round) => (
      <>
        Slime pauses each engine through its own Service on port 30000, has it load the checkpoint with{" "}
        <code>/update_weights_from_disk</code>, and resumes it. The frontend isn't on this path. The next round
        generates with weights v{round + 1}.
      </>
    ),
  },
];

const PACKETS: Record<
  StepId,
  ReadonlyArray<{ wire: WireId; tone: Tone | "return"; reverse?: boolean; at: readonly number[] }>
> = {
  generate: [
    { wire: "generate", tone: "green", at: [0, 0.6, 1.2] },
    { wire: "route0", tone: "green", at: [0.3, 1.2] },
    { wire: "route1", tone: "green", at: [0.75, 1.65] },
    { wire: "route0", tone: "return", reverse: true, at: [0.9] },
    { wire: "route1", tone: "return", reverse: true, at: [0.15] },
    { wire: "generate", tone: "return", reverse: true, at: [0.45, 1.35] },
  ],
  score: [],
  train: [{ wire: "batch", tone: "amber", at: [0, 0.6, 1.2] }],
  write: [{ wire: "write", tone: "amber", at: [0, 0.6, 1.2] }],
  reload: [
    { wire: "control", tone: "amber", at: [0, 0.9] },
    { wire: "load", tone: "green", at: [0.3, 1.2] },
  ],
};

/** Illustrative rewards for one prompt's 8 samples: 1 is a correct answer. */
const REWARDS = [1, 0, 1, 1, 0, 1, 1, 0] as const;

const chipWidth = (text: string) => Math.round(text.length * 6.3 + 16);

export function SlimeIntegrationMap() {
  const [round, setRound] = useState(1);
  // -1 before the round starts, 0..4 on a step, STEPS.length once the round is done.
  const [phase, setPhase] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [focus, setFocus] = useState<Owner | null>(null);

  useEffect(() => {
    if (!playing) return;
    if (phase >= STEPS.length) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setPhase(phase + 1), STEP_MS);
    return () => window.clearTimeout(timer);
  }, [playing, phase]);

  useEffect(() => {
    if (!playing) return;
    const pauseWhenHidden = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () => document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, [playing]);

  const step = phase >= 0 && phase < STEPS.length ? STEPS[phase] : null;
  const done = phase >= STEPS.length;
  const lastRound = round >= ROUNDS;

  const play = () => {
    if (playing) {
      setPlaying(false);
    } else if (done && lastRound) {
      setRound(1);
      setPhase(-1);
    } else if (done) {
      setRound(round + 1);
      setPhase(0);
      setPlaying(true);
    } else {
      if (phase < 0) setPhase(0);
      setPlaying(true);
    }
  };

  const goTo = (index: number) => {
    setPlaying(false);
    setPhase(index);
  };

  const playLabel = playing
    ? "Pause"
    : done
      ? lastRound
        ? "Start over"
        : `Play round ${round + 1}`
      : phase >= 0
        ? "Play from here"
        : "Play a round";
  const playIcon = playing ? "❚❚" : done && lastRound ? "↺" : "▶";

  let title: ReactNode;
  let text: ReactNode;
  if (step) {
    title = `${phase + 1}. ${step.label}`;
    text = step.text(round);
  } else if (done) {
    title = `Round ${round} done`;
    text = lastRound
      ? `That was the last of the tutorial's ${ROUNDS} rounds. The RayJob succeeds, and KubeRay deletes the Ray cluster. The DynamoGraphDeployment keeps serving until you delete it.`
      : `Both engines now serve weights v${round + 1}, and Slime deletes the checkpoint once they load it. Play the next round, or pick a step to replay this one.`;
  } else {
    title = "Follow one round";
    text = "Press Play a round to follow one GRPO step of the tutorial through these parts, or pick a step.";
  }

  const nodeState = (id: NodeId, owner: Owner) => {
    if (focus) return owner === focus ? " is-focus" : " is-dim";
    if (!step) return "";
    return step.nodes.includes(id) ? " is-active" : " is-dim";
  };

  const wireState = (id: WireId) => {
    if (focus) return " is-quiet";
    if (!step) return "";
    return step.wires.includes(id) ? " is-active" : " is-quiet";
  };

  const reloading = step?.id === "reload";
  const version = reloading ? `loading v${round + 1}` : `weights v${done ? round + 1 : round}`;
  const showRewards = step?.id === "score" || step?.id === "train";
  const weightChip =
    step?.id === "write" || reloading ? `weight-updates/${checkpoint(round + 1)}/` : "weight-updates/";
  const chips = [
    { text: "models/Qwen3-0.6B/", isNew: false },
    { text: "datasets/gsm8k/", isNew: false },
    { text: weightChip, isNew: step?.id === "write" },
  ];

  return (
    <div className="slmap not-prose">
      <SlimeIntegrationMapStyles />

      <div className="slmap-owners">
        {OWNERS.map((owner) => (
          <article
            key={owner.id}
            className={`slmap-owner slmap-owner--${owner.id}`}
            onMouseEnter={() => setFocus(owner.id)}
            onMouseLeave={() => setFocus(null)}
          >
            <p className="slmap-owner__eyebrow">
              <span className="slmap-owner__num">{owner.num}</span>
              {owner.role}
            </p>
            <h3 className="slmap-owner__name">{OWNER_NAMES[owner.id]}</h3>
            <p className="slmap-owner__text">{owner.text}</p>
            <div className="slmap-flow">
              <span>{owner.flow[0]}</span>
              <b aria-hidden="true">{owner.flow[1]}</b>
              <span>{owner.flow[2]}</span>
            </div>
          </article>
        ))}
      </div>
      <p className="slmap-caption">
        Rollouts go through the Dynamo frontend. Slime sends control calls straight to each engine, and the new
        weights move through the shared volume.
      </p>

      <section className="slmap-world" aria-label="One training round">
        <div className="slmap-world__head">
          <div>
            <p className="slmap-eyebrow">One training round</p>
            <p className="slmap-round">
              Round {round} of {ROUNDS}
            </p>
          </div>
          <button type="button" className="slmap-play" onClick={play}>
            <span className="slmap-play__icon" aria-hidden="true">
              {playIcon}
            </span>
            {playLabel}
          </button>
        </div>

        <ol className="slmap-steps" aria-label="Steps in a round">
          {STEPS.map((item, index) => (
            <li key={item.id}>
              <button
                type="button"
                className={`slmap-step${phase > index ? " is-done" : ""}`}
                aria-current={phase === index ? "step" : undefined}
                onClick={() => goTo(index)}
              >
                <span className="slmap-step__num" aria-hidden="true">
                  {phase > index ? "✓" : index + 1}
                </span>
                {item.label}
              </button>
            </li>
          ))}
        </ol>

        <div className="slmap-stage">
          <svg
            className="slmap-svg"
            viewBox="0 0 800 398"
            role="img"
            aria-label="The Ray head Pod runs Slime and its Megatron-LM trainer. The DynamoGraphDeployment runs the Dynamo frontend and two SGLang engines. Slime sends rollouts through the frontend, sends control calls straight to each engine, and passes new weights through the shared volume."
          >
            <rect
              className={`slmap-zone slmap-zone--ray${focus === "kuberay" ? " is-focus" : ""}`}
              x={12}
              y={34}
              width={256}
              height={262}
              rx={14}
            />
            <text className="slmap-zone-label slmap-zone-label--ray" x={28} y={57}>
              RAY HEAD POD
            </text>
            <text className="slmap-zone-owner" x={254} y={57} textAnchor="end">
              KubeRay
            </text>

            <rect
              className={`slmap-zone slmap-zone--dgd${focus === "dynamo" ? " is-focus" : ""}`}
              x={340}
              y={34}
              width={448}
              height={262}
              rx={14}
            />
            <text className="slmap-zone-label slmap-zone-label--dgd" x={356} y={57}>
              DYNAMOGRAPHDEPLOYMENT
            </text>
            <text className="slmap-zone-owner" x={774} y={57} textAnchor="end">
              Dynamo operator
            </text>

            <rect className="slmap-zone slmap-zone--engines" x={552} y={70} width={222} height={214} rx={12} />
            <text className="slmap-zone-label slmap-zone-label--engines" x={566} y={88}>
              SGLANG ENGINES
            </text>

            <rect
              className={`slmap-zone slmap-zone--volume${step?.volume ? " is-active" : ""}`}
              x={12}
              y={326}
              width={776}
              height={62}
              rx={12}
            />
            <text className="slmap-zone-label slmap-zone-label--volume" x={28} y={349}>
              SHARED VOLUME · /data
            </text>
            <text className="slmap-zone-owner" x={772} y={349} textAnchor="end">
              slime-data PVC
            </text>
            {chips.reduce<{ x: number; nodes: ReactNode[] }>(
              (row, chip) => {
                const width = chipWidth(chip.text);
                row.nodes.push(
                  <g key={chip.text} className={`slmap-chip${chip.isNew ? " is-new" : ""}`}>
                    <rect x={row.x} y={358} width={width} height={18} rx={5} />
                    <text x={row.x + 8} y={370.5}>
                      {chip.text}
                    </text>
                  </g>,
                );
                row.x += width + 8;
                return row;
              },
              { x: 28, nodes: [] },
            ).nodes}

            {WIRES.map((wire) => {
              const state = wireState(wire.id);
              const tone = ` slmap-tone--${wire.tone}`;
              return (
                <g key={wire.id}>
                  <path
                    className={`slmap-wire${wire.dashed ? " slmap-wire--control" : ""}${tone}${state}`}
                    d={wire.d}
                  />
                  <polygon className={`slmap-head${tone}${state}`} points={wire.head} />
                </g>
              );
            })}

            {NODES.map((node) => (
              <g key={node.id} className={`slmap-node${nodeState(node.id, node.owner)}`} data-owner={node.owner}>
                <rect className="slmap-node__halo" x={node.x} y={node.y} width={node.w} height={node.h} rx={10} />
                <rect className="slmap-node__box" x={node.x} y={node.y} width={node.w} height={node.h} rx={10} />
                <text className="slmap-node__title" x={node.x + 12} y={node.y + 23}>
                  {node.title}
                </text>
                {node.sub.map((line, index) => (
                  <text key={line} className="slmap-node__sub" x={node.x + 12} y={node.y + 40 + index * 16}>
                    {line}
                  </text>
                ))}
                {node.id === "slime" && showRewards
                  ? REWARDS.map((reward, index) => (
                      <circle
                        key={index}
                        className={`slmap-reward slmap-reward--${reward ? "hit" : "miss"}`}
                        cx={node.x + 17 + index * 15}
                        cy={node.y + 61}
                        r={5}
                        style={{ animationDelay: `${index * 90}ms` }}
                      />
                    ))
                  : null}
                {node.id === "trainer" && step?.id === "train" ? (
                  <>
                    <rect className="slmap-progress-track" x={node.x + 12} y={node.y + 56} width={node.w - 24} height={6} rx={3} />
                    <rect
                      key={round}
                      className="slmap-progress"
                      x={node.x + 12}
                      y={node.y + 56}
                      width={node.w - 24}
                      height={6}
                      rx={3}
                    />
                  </>
                ) : null}
                {node.id === "engine0" || node.id === "engine1" ? (
                  <>
                    <g className={`slmap-version${reloading ? " is-loading" : ""}`}>
                      <rect x={node.x + 12} y={node.y + 47} width={chipWidth(version) - 4} height={17} rx={8.5} />
                      <text x={node.x + 18} y={node.y + 59}>
                        {version}
                      </text>
                    </g>
                    <text className="slmap-node__sub" x={node.x + node.w - 12} y={node.y + 59} textAnchor="end">
                      1 GPU
                    </text>
                  </>
                ) : null}
              </g>
            ))}

            {WIRES.map((wire) =>
              wire.label ? (
                <text
                  key={wire.id}
                  className={`slmap-wire-label slmap-wire-label--on-${wire.label.on} slmap-tone--${wire.tone}${wireState(wire.id)}`}
                  x={wire.label.x}
                  y={wire.label.y}
                  textAnchor={wire.label.anchor}
                >
                  {wire.label.text}
                </text>
              ) : null,
            )}

            {step && !focus
              ? PACKETS[step.id].flatMap((packet) =>
                  packet.at.map((at) => (
                    <circle
                      key={`${step.id}-${packet.wire}-${packet.tone}-${at}`}
                      className={`slmap-packet slmap-packet--${packet.tone}`}
                      r={4.2}
                    >
                      <animateMotion
                        dur={`${TRAVEL_S}s`}
                        begin={at ? `-${at}s` : "0s"}
                        repeatCount="indefinite"
                        path={WIRE_PATHS[packet.wire]}
                        keyPoints={packet.reverse ? "1;0" : "0;1"}
                        keyTimes="0;1"
                        calcMode="linear"
                      />
                    </circle>
                  )),
                )
              : null}
          </svg>
        </div>

        <div className="slmap-status" aria-live="polite">
          <p className="slmap-status__title">
            {title}
            {(step?.owners ?? []).map((owner) => (
              <span key={owner} className={`slmap-tag slmap-tag--${owner}`}>
                {OWNER_NAMES[owner]}
              </span>
            ))}
          </p>
          <p className="slmap-status__text">{text}</p>
        </div>
        <p className="slmap-footnote">
          Simplified from the tutorial's validated run: {ROUNDS} rounds on three GPUs. Not drawn to time.
        </p>
      </section>
    </div>
  );
}

export default SlimeIntegrationMap;
