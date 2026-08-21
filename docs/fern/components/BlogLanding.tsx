/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const ARTICLES = [
  {
    title: "Dynamo Agent Optimization Skills",
    description:
      "A skillpack that turns coding agents into disciplined performance engineers: objective functions, experimental discipline, and inference domain knowledge.",
    href: "/dynamo/dev/digest/agent-optimization-skills",
    date: "August 21, 2026",
    readTime: "2 min read",
    category: "Agentic AI",
    art: "hillclimb",
  },
  {
    title: "DynoSim: Simulating the Pareto Frontier",
    description:
      "Explore serving configurations with a workload-driven Dynamo simulator before committing scarce GPU time to cluster validation.",
    href: "/dynamo/dev/digest/dynosim-pareto-frontier",
    date: "May 29, 2026",
    readTime: "1 min read",
    category: "Simulation",
    art: "frontier",
  },
  {
    title: "NVIDIA Dynamo Snapshot: Fast Startup for Inference Workloads on Kubernetes",
    description:
      "See how checkpoint and restore techniques bring warm inference workers online in seconds instead of minutes.",
    href: "/dynamo/dev/digest/dynamo-snapshot-fast-startup",
    date: "May 28, 2026",
    readTime: "1 min read",
    category: "Kubernetes",
    art: "snapshot",
  },
  {
    title: "Dynamo Day 0 Support for TokenSpeed",
    description:
      "A launch note on TokenSpeed, its scheduler and kernel work, and the first Dynamo backend integration.",
    href: "/dynamo/dev/digest/tokenspeed-day-0",
    date: "May 6, 2026",
    readTime: "1 min read",
    category: "Ecosystem",
    art: "tokens",
  },
  {
    title: "Streaming Tokens and Tools: Multi-Turn Agentic Harness Support in Dynamo",
    description:
      "Lessons from running Claude Code, Codex, and OpenClaw against Dynamo, from prompt stability to streaming tool dispatch.",
    href: "/dynamo/dev/digest/agentic-harnesses",
    date: "April 30, 2026",
    readTime: "18 min read",
    category: "Agentic AI",
    art: "agents",
  },
  {
    title: "Full-Stack Optimizations for Agentic Inference with Dynamo",
    description:
      "How the frontend API, KV router, and cache-management layers work together for long-running agentic workloads.",
    href: "/dynamo/dev/digest/agentic-inference",
    date: "March 2026",
    readTime: "16 min read",
    category: "Architecture",
    art: "stack",
  },
  {
    title: "Flash Indexer: A Story of Inter-Galactic KV Routing",
    description:
      "The six design iterations behind a concurrent global KV index capable of sustaining more than 100 million operations per second.",
    href: "/dynamo/dev/digest/flash-indexer",
    date: "February 23, 2026",
    readTime: "13 min read",
    category: "Engineering",
    art: "indexer",
  },
] as const;

function ArrowIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M4 10h11M11 5l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M11 4h5v5M16 4l-7 7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M15 11v4a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ArticleArt({ variant }: { variant: string }) {
  return (
    <div className={`dynamo-blog-art dynamo-blog-art--${variant}`} aria-hidden="true">
      <span className="dynamo-blog-art__grid" />
      <span className="dynamo-blog-art__orb dynamo-blog-art__orb--one" />
      <span className="dynamo-blog-art__orb dynamo-blog-art__orb--two" />
      {variant === "hillclimb" ? (
        <svg className="dynamo-blog-art__climb" viewBox="0 0 200 120" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M10 96 H40 V72 H68 V58 H96 V52 H128 V30 H176"
            stroke="rgba(255,255,255,0.9)"
            strokeWidth="2.4"
            strokeLinecap="square"
          />
          <circle cx="40" cy="72" r="3.4" fill="#76b900" />
          <circle cx="68" cy="58" r="3.4" fill="#76b900" />
          <circle cx="96" cy="52" r="3.4" fill="#76b900" />
          <circle cx="128" cy="30" r="3.4" fill="#76b900" />
          <circle cx="150" cy="38" r="2.6" fill="rgba(255,255,255,0.45)" />
          <circle cx="82" cy="80" r="2.6" fill="rgba(255,255,255,0.45)" />
          <circle cx="176" cy="30" r="5.4" stroke="#76b900" strokeWidth="2" fill="none" />
          <path d="M22 104 l6 6 m0 -6 l-6 6" stroke="rgba(255,130,130,0.85)" strokeWidth="2" strokeLinecap="round" />
          <path d="M50 104 l6 6 m0 -6 l-6 6" stroke="rgba(255,130,130,0.85)" strokeWidth="2" strokeLinecap="round" />
        </svg>
      ) : (
        <>
          <span className="dynamo-blog-art__line dynamo-blog-art__line--one" />
          <span className="dynamo-blog-art__line dynamo-blog-art__line--two" />
        </>
      )}
      <span className="dynamo-blog-art__mark">D</span>
    </div>
  );
}

export function BlogLanding() {
  return (
    <div className="dynamo-blog-home">
      <section className="dynamo-blog-latest" id="latest" aria-labelledby="latest-heading">
        <div className="dynamo-blog-section-heading">
          <div className="dynamo-blog-section-heading__copy">
            <span className="dynamo-blog-kicker">From the team</span>
            <h2 id="latest-heading">Latest articles</h2>
            <p>Technical perspectives on distributed inference, performance, and the Dynamo ecosystem.</p>
          </div>
          <div className="dynamo-blog-section-heading__actions">
            <a
              className="dynamo-blog-button dynamo-blog-button--secondary"
              href="https://github.com/ai-dynamo/dynamo/subscription"
              target="_blank"
              rel="noopener noreferrer"
            >
              Subscribe on GitHub
              <ExternalLinkIcon />
            </a>
          </div>
        </div>

        <div className="dynamo-blog-grid">
          {ARTICLES.map((article) => (
            <article className="dynamo-blog-card" key={article.href}>
              <a className="dynamo-blog-card__art-link" href={article.href} tabIndex={-1} aria-hidden="true">
                <ArticleArt variant={article.art} />
              </a>
              <div className="dynamo-blog-card__body">
                <div className="dynamo-blog-card__topline">
                  <span>{article.category}</span>
                  <span>{article.readTime}</span>
                </div>
                <h3><a href={article.href}>{article.title}</a></h3>
                <p>{article.description}</p>
                <div className="dynamo-blog-card__footer">
                  <time>{article.date}</time>
                  <a href={article.href} aria-label={`Read ${article.title}`}>
                    <ArrowIcon />
                  </a>
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
