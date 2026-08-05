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
import { UPCOMING_EVENTS } from "./events.generated";

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

function SlackIcon() {
  return (
    <svg viewBox="0 0 448 512" aria-hidden="true">
      <path d="M94.1 315.1c0 25.9-21.2 47.1-47.1 47.1S0 341 0 315.1 21.2 268 47.1 268h47.1v47.1Zm23.7 0c0-25.9 21.2-47.1 47.1-47.1s47.1 21.2 47.1 47.1v117.8c0 25.9-21.2 47.1-47.1 47.1s-47.1-21.2-47.1-47.1V315.1Zm47.1-189c-25.9 0-47.1-21.2-47.1-47.1S139 32 164.9 32 212 53.2 212 79.1v47.1h-47.1Zm0 23.7c25.9 0 47.1 21.2 47.1 47.1S190.8 244 164.9 244H47.1C21.2 244 0 222.8 0 196.9s21.2-47.1 47.1-47.1h117.8Zm189 47.1c0-25.9 21.2-47.1 47.1-47.1s47.1 21.2 47.1 47.1-21.2 47.1-47.1 47.1h-47.1v-47.1Zm-23.7 0c0 25.9-21.2 47.1-47.1 47.1S236 222.8 236 196.9V79.1C236 53.2 257.2 32 283.1 32s47.1 21.2 47.1 47.1v117.8Zm-47.1 189c25.9 0 47.1 21.2 47.1 47.1S309 480 283.1 480 236 458.8 236 432.9v-47.1h47.1Zm0-23.7c-25.9 0-47.1-21.2-47.1-47.1s21.2-47.1 47.1-47.1h117.8c25.9 0 47.1 21.2 47.1 47.1s-21.2 47.1-47.1 47.1H283.1Z" />
    </svg>
  );
}

function CalendarAppIcon() {
  const day = UPCOMING_EVENTS[0]?.day ?? "•";
  return (
    <span className="dynamo-welcome__calendar-app" aria-hidden="true">
      <span>CAL</span>
      <strong>{day}</strong>
    </span>
  );
}

function CommunityRail() {
  const notifications = [
    {
      app: "AI Dynamo Slack",
      message: "Join the Dynamo community",
      href: SLACK_URL,
      icon: <SlackIcon />,
      tone: "slack",
    },
    {
      app: "Calendar",
      message: "See upcoming community events",
      href: CALENDAR_URL,
      icon: <CalendarAppIcon />,
      tone: "calendar",
    },
  ];

  return (
    <nav
      className="dynamo-welcome__community"
      aria-label="Dynamo community links"
    >
      {notifications.map(({ app, message, href, icon, tone }) => (
        <a
          key={app}
          className={`dynamo-welcome__notification dynamo-welcome__notification--${tone}`}
          href={href}
          target={href.startsWith("http") ? "_blank" : undefined}
          rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
        >
          <span className="dynamo-welcome__notification-icon">{icon}</span>
          <span className="dynamo-welcome__notification-copy">
            <span className="dynamo-welcome__notification-app">
              {app}
              <small>now</small>
            </span>
            <span>{message}</span>
          </span>
        </a>
      ))}
    </nav>
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

      <CommunityRail />
    </div>
  );
}

export default WelcomeHero;
