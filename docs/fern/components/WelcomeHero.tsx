/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * WelcomeHero — interactive continuation of the Welcome page heading.
 *
 * Fern renders the page title and subtitle from frontmatter. This component
 * starts immediately below them with a rotating Dynamo statement, the primary
 * quickstart action, a community notification stack, and the terminal demonstration.
 */
"use client";

import { useEffect, useRef, useState } from "react";
import { TerminalDemo } from "./TerminalDemo";

const STATEMENTS = [
  "works with vLLM, SGLang, and TensorRT-LLM.",
  "supports NVIDIA and AMD GPUs, and Intel XPUs.",
  "runs on Kubernetes, Slurm, or locally.",
  "scales one component or the full stack.",
];

const CALENDAR_URL =
  "https://calendar.google.com/calendar/u/0/r?cid=Y19jMjQ0OGQyZWZiMDllYWMyZGRlZTFmMzQ1MjQxMjQxMzViZDNmNDU1NDg2ODc2OTA1OTEwNWUxOGUxYjk3ZThmQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20";
const SLACK_URL = "http://ai-dynamo.org/slack";

function RotatingStatement() {
  const [statementIndex, setStatementIndex] = useState(0);
  const [displayed, setDisplayed] = useState(STATEMENTS[0]);
  const [deleting, setDeleting] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updatePreference = () => setReduceMotion(query.matches);
    updatePreference();
    query.addEventListener("change", updatePreference);
    return () => query.removeEventListener("change", updatePreference);
  }, []);

  useEffect(() => {
    if (reduceMotion) {
      setStatementIndex(0);
      setDisplayed(STATEMENTS[0]);
      setDeleting(false);
      return;
    }

    const statement = STATEMENTS[statementIndex];
    let delay = deleting ? 34 : 62;
    let next = displayed;
    let nextDeleting = deleting;
    let nextIndex = statementIndex;

    if (!deleting && displayed === statement) {
      delay = 1800;
      nextDeleting = true;
    } else if (deleting && displayed.length === 0) {
      delay = 260;
      nextDeleting = false;
      nextIndex = (statementIndex + 1) % STATEMENTS.length;
    } else if (deleting) {
      next = statement.slice(0, Math.max(0, displayed.length - 1));
    } else {
      next = statement.slice(0, displayed.length + 1);
    }

    const timer = window.setTimeout(() => {
      setDisplayed(next);
      setDeleting(nextDeleting);
      setStatementIndex(nextIndex);
    }, delay);

    return () => window.clearTimeout(timer);
  }, [deleting, displayed, reduceMotion, statementIndex]);

  return (
    <div className="dynamo-welcome__statement">
      <p aria-hidden="true">
        <span>Dynamo </span>
        <span className="dynamo-welcome__typed">{displayed}</span>
        <span className="dynamo-welcome__cursor" />
      </p>
      <p className="dynamo-welcome__sr-only">
        Dynamo works with vLLM, SGLang, and TensorRT-LLM; supports NVIDIA and
        AMD GPUs and Intel XPUs; runs on Kubernetes, Slurm, or locally; and
        scales one component or the full stack.
      </p>
    </div>
  );
}

export interface WelcomeHeroProps {
  /** Fern-rewritten path to the asciinema recording. */
  src: string;
}

export function WelcomeHero({ src }: WelcomeHeroProps) {
  const demoRef = useRef<HTMLElement | null>(null);
  const [demoVisible, setDemoVisible] = useState(false);

  useEffect(() => {
    const demo = demoRef.current;
    if (!demo) return;

    let frame = 0;
    const updateVisibility = () => {
      frame = 0;
      const top = demo.getBoundingClientRect().top;
      setDemoVisible(window.scrollY > 80 && top < window.innerHeight * 0.82);
    };
    const onScroll = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(updateVisibility);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    updateVisibility();

    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  return (
    <div className="dynamo-welcome">
      <section
        className="dynamo-welcome__intro"
        aria-label="Get started with Dynamo"
      >
        <RotatingStatement />
        <div className="dynamo-welcome__actions">
          <a className="dynamo-welcome__cta" href="/dynamo/dev/kubernetes">
            Get started
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="m9 18 6-6-6-6" />
            </svg>
          </a>
          <a className="dynamo-welcome__cta dynamo-welcome__cta--secondary" href={SLACK_URL} target="_blank" rel="noopener noreferrer">
            Join Slack
          </a>
          <a className="dynamo-welcome__cta dynamo-welcome__cta--secondary" href={CALENDAR_URL} target="_blank" rel="noopener noreferrer">
            View calendar
          </a>
        </div>
      </section>

      <section
        ref={demoRef}
        className="dynamo-welcome__terminal"
        data-visible={demoVisible}
        aria-labelledby="dynamo-demo-heading"
      >
        <div className="dynamo-welcome__demo-reveal">
          <p>Deployment walkthrough</p>
          <h2 id="dynamo-demo-heading">See Dynamo in action</h2>
        </div>
        <div className="dynamo-welcome__terminal-stage">
          <TerminalDemo
            src={src}
            title="Qwen3-235B deployment"
            idleTimeLimit={1.5}
            speed={1.0}
            rows="25"
          />
        </div>
      </section>
    </div>
  );
}

export default WelcomeHero;
