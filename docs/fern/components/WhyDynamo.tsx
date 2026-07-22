/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * WhyDynamo — scroll-driven feature story for the Welcome page.
 */
"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

const FEATURES = [
  {
    eyebrow: "Distributed inference",
    title: "Scale the system around the model",
    description:
      "Disaggregated serving, KV-aware routing, cache management, and autoscaling coordinate the full inference pipeline.",
    graphic: "performance",
  },
  {
    eyebrow: "Engine interoperability",
    title: "Bring your inference engine",
    description:
      "Use Dynamo with vLLM, SGLang, or TensorRT-LLM, then connect the network, storage, and KV cache layers you need.",
    graphic: "engines",
  },
  {
    eyebrow: "Infrastructure choice",
    title: "Run where your workloads run",
    description:
      "Deploy on Kubernetes, schedule with Slurm, or start locally across NVIDIA and AMD GPUs and Intel XPUs.",
    graphic: "infrastructure",
  },
  {
    eyebrow: "Modular adoption",
    title: "Adopt one component or the full stack",
    description:
      "Start with the frontend, router, planner, or cache manager, and add the rest as your deployment grows.",
    graphic: "modular",
  },
] as const;

function PerformanceGraphic() {
  return (
    <div className="dynamo-story-graphic dynamo-story-graphic--performance">
      <div className="dynamo-story-windowbar">
        <span />
        <span />
        <span />
        <code>dynamo metrics</code>
      </div>
      <div className="dynamo-story-metrics">
        <div>
          <span>Time to first token</span>
          <strong>↓</strong>
          <i style={{ "--metric": "72%" } as CSSProperties} />
        </div>
        <div>
          <span>Request throughput</span>
          <strong>↑</strong>
          <i style={{ "--metric": "91%" } as CSSProperties} />
        </div>
        <div>
          <span>KV cache reuse</span>
          <strong>↑</strong>
          <i style={{ "--metric": "84%" } as CSSProperties} />
        </div>
      </div>
      <div className="dynamo-story-flow">
        <span>Prefill</span>
        <b>KV-aware router</b>
        <span>Decode</span>
      </div>
    </div>
  );
}

function EnginesGraphic() {
  return (
    <div className="dynamo-story-graphic dynamo-story-graphic--engines">
      <div className="dynamo-story-windowbar">
        <span />
        <span />
        <span />
        <code>dynamo serve</code>
      </div>
      <pre>
        <span className="prompt">$</span> dynamo serve --backend <b>vllm</b>
        {"\n"}
        <span className="muted">✓ frontend ready</span>
        {"\n"}
        <span className="muted">✓ router connected</span>
        {"\n"}
        <span className="muted">✓ workers discovered</span>
        {"\n\n"}
        <span className="success">Serving on :8000</span>
      </pre>
      <div className="dynamo-story-engine-tabs">
        <span>vLLM</span>
        <span>SGLang</span>
        <span>TensorRT-LLM</span>
      </div>
    </div>
  );
}

function InfrastructureGraphic() {
  return (
    <div className="dynamo-story-graphic dynamo-story-graphic--infrastructure">
      <div className="dynamo-story-orbit">
        <div className="dynamo-story-core">Dynamo</div>
        <span className="node node--k8s">Kubernetes</span>
        <span className="node node--slurm">Slurm</span>
        <span className="node node--local">Local</span>
      </div>
      <div className="dynamo-story-hardware">
        <span>NVIDIA GPU</span>
        <span>AMD GPU</span>
        <span>Intel XPU</span>
      </div>
    </div>
  );
}

function ModularGraphic() {
  return (
    <div className="dynamo-story-graphic dynamo-story-graphic--modular">
      <div className="dynamo-story-stack">
        <span>Frontend</span>
        <span>KV router</span>
        <span>Planner</span>
        <span>KV cache manager</span>
        <span>Workers</span>
      </div>
      <div className="dynamo-story-stack-caption">
        <span>Start small</span>
        <i />
        <span>Scale out</span>
      </div>
    </div>
  );
}

const GRAPHICS = {
  performance: <PerformanceGraphic />,
  engines: <EnginesGraphic />,
  infrastructure: <InfrastructureGraphic />,
  modular: <ModularGraphic />,
};

export function WhyDynamo() {
  const [activeIndex, setActiveIndex] = useState(0);
  const steps = useRef<Array<HTMLElement | null>>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const index = Number((visible.target as HTMLElement).dataset.index);
        if (!Number.isNaN(index)) setActiveIndex(index);
      },
      { rootMargin: "-28% 0px -42%", threshold: [0.15, 0.35, 0.6] },
    );

    steps.current.forEach((step) => step && observer.observe(step));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="dynamo-story">
      <div className="dynamo-story__steps">
        {FEATURES.map((feature, index) => (
          <section
            key={feature.title}
            ref={(node) => {
              steps.current[index] = node;
            }}
            data-index={index}
            data-active={activeIndex === index}
            className="dynamo-story__step"
            aria-labelledby={`dynamo-story-title-${index}`}
          >
            <div className="dynamo-story__step-copy">
              <p className="dynamo-story__eyebrow">
                {String(index + 1).padStart(2, "0")} · {feature.eyebrow}
              </p>
              <h3 id={`dynamo-story-title-${index}`}>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
            <div className="dynamo-story__mobile-graphic" aria-hidden="true">
              {GRAPHICS[feature.graphic]}
            </div>
          </section>
        ))}
      </div>

      <div className="dynamo-story__stage" aria-hidden="true">
        {FEATURES.map((feature, index) => (
          <div
            key={feature.graphic}
            className="dynamo-story__stage-panel"
            data-active={activeIndex === index}
          >
            {GRAPHICS[feature.graphic]}
          </div>
        ))}
        <div className="dynamo-story__progress" aria-hidden="true">
          {FEATURES.map((feature, index) => (
            <span key={feature.title} data-active={activeIndex === index} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default WhyDynamo;
