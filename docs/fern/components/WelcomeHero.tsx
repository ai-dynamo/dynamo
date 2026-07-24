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

import { useEffect, useState } from "react";
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

function WeChatIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8.691 2.188C3.891 2.188 0 5.476 0 9.53c0 2.212 1.17 4.203 3.002 5.55a.59.59 0 0 1 .213.665l-.39 1.48c-.019.07-.048.141-.048.213 0 .163.13.295.29.295a.326.326 0 0 0 .167-.054l1.903-1.114a.864.864 0 0 1 .717-.098 10.16 10.16 0 0 0 2.837.403c.276 0 .543-.027.811-.05-.857-2.578.157-4.972 1.932-6.446 1.703-1.415 3.882-1.98 5.853-1.838-.576-3.583-4.196-6.348-8.596-6.348ZM5.785 5.991c.642 0 1.162.529 1.162 1.18a1.17 1.17 0 0 1-1.162 1.178A1.17 1.17 0 0 1 4.623 7.17c0-.651.52-1.18 1.162-1.18Zm5.813 0c.642 0 1.162.529 1.162 1.18a1.17 1.17 0 0 1-1.162 1.178 1.17 1.17 0 0 1-1.162-1.178c0-.651.52-1.18 1.162-1.18Zm5.34 2.867c-1.797-.052-3.746.512-5.28 1.786-1.72 1.428-2.687 3.72-1.78 6.22.942 2.453 3.666 4.229 6.884 4.229.826 0 1.622-.12 2.361-.336a.722.722 0 0 1 .598.082l1.584.926a.272.272 0 0 0 .14.047c.134 0 .24-.111.24-.247 0-.06-.023-.12-.038-.177l-.327-1.233a.582.582 0 0 1-.023-.156.49.49 0 0 1 .201-.398C23.024 18.48 24 16.82 24 14.98c0-3.21-2.931-5.837-6.656-6.088v-.002c-.135-.01-.27-.027-.407-.03Zm-2.53 3.274c.535 0 .969.44.969.982a.976.976 0 0 1-.969.983.976.976 0 0 1-.969-.983c0-.542.434-.982.97-.982Zm4.844 0c.535 0 .969.44.969.982a.976.976 0 0 1-.969.983.976.976 0 0 1-.969-.983c0-.542.434-.982.969-.982Z" />
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
      app: "CNCF Slack",
      message: "Join the #ai-dynamo channel",
      href: "https://communityinviter.com/apps/cloud-native/cncf",
      icon: <SlackIcon />,
      tone: "slack",
    },
    {
      app: "WeChat",
      message: "Ask to join the Dynamo community group",
      href: "/dynamo/dev/contribution-guide.zh-CN",
      icon: <WeChatIcon />,
      tone: "wechat",
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
  return (
    <div className="dynamo-welcome">
      <section
        className="dynamo-welcome__intro"
        aria-label="Get started with Dynamo"
      >
        <RotatingStatement />
        <a className="dynamo-welcome__cta" href="/dynamo/dev/kubernetes">
          Get started
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m9 18 6-6-6-6" />
          </svg>
        </a>
      </section>

      <div className="dynamo-welcome__terminal">
        <TerminalDemo
          src={src}
          title="Deploy Qwen3-235B with Dynamo"
          idleTimeLimit={1.5}
          speed={1.0}
          rows="25"
        />
      </div>

      <CommunityRail />
    </div>
  );
}

export default WelcomeHero;
