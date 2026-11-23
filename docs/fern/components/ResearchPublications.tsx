/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Research publications: deep dives (papers that use, extend, or benchmark
 * Dynamo) and citations (papers that only cite it). A view toggle picks one
 * set or both, one keyword filter applies to whatever is shown, and citations
 * page with "Show more".
 *
 * Same card as the partner list next door, with one swap: papers have no
 * publisher logo worth showing, so the venue takes the tile's place. It is
 * also the more useful signal on a paper.
 *
 * Card styles live in PublicationsStyles.tsx, shared with EcosystemPublications;
 * a page rendering this must render <BlogStyles /> and <PublicationsStyles />.
 */
"use client";

import { useState } from "react";
import {
  CITING_PAPERS,
  RESEARCH_PAPERS,
  RESEARCH_TAGS,
  type ResearchPaper,
  type ResearchTag,
} from "./research-papers.data";
import { ExternalMark } from "./PublicationsStyles";

function ResearchCard({ paper }: { paper: ResearchPaper }) {
  const { title, url, org, venue, date } = paper;
  const words = title.trim().split(/\s+/);
  const lastWord = words.pop() ?? "";
  const leadingWords = words.join(" ");
  return (
    <a
      className="dynamo-pubs__card"
      href={url}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="dynamo-pubs__top">
        <span className="dynamo-pubs__venue">{venue}</span>
        <span className="dynamo-pubs__partner">{org}</span>
        <span className="dynamo-pubs__date">{date}</span>
      </span>
      <span className="dynamo-pubs__title">
        {leadingWords ? `${leadingWords} ` : ""}
        <span className="dynamo-pubs__title-end">
          {lastWord}
          <ExternalMark />
        </span>
      </span>
      <span className="dynamo-pubs__tags">{paper.tags.join(" · ")}</span>
    </a>
  );
}

// Newest first. Sorted from `iso` rather than trusting the array's order, so a
// new entry can be appended anywhere and still land in the right place.
const byDate = (a: ResearchPaper, b: ResearchPaper) => b.iso.localeCompare(a.iso);
const sortedPapers = [...RESEARCH_PAPERS].sort(byDate);
const sortedCitations = [...CITING_PAPERS].sort(byDate);

const CITATION_PAGE = 12;

type View = "deep" | "citations" | "all";

export function ResearchPublications() {
  const [view, setView] = useState<View>("deep");
  const [activeTag, setActiveTag] = useState<ResearchTag | null>(null);
  const [citationsShown, setCitationsShown] = useState(CITATION_PAGE);

  const inView = [
    ...(view !== "citations" ? RESEARCH_PAPERS : []),
    ...(view !== "deep" ? CITING_PAPERS : []),
  ];
  // Counts follow the selected view, and a keyword with nothing in it drops out.
  const tagCounts = RESEARCH_TAGS.map((tag) => ({
    tag,
    count: inView.filter((p) => p.tags.includes(tag)).length,
  })).filter(({ count }) => count > 0);

  const matches = (p: ResearchPaper) => !activeTag || p.tags.includes(activeTag);
  const deepDives = view !== "citations" ? sortedPapers.filter(matches) : [];
  const citationMatches = view !== "deep" ? sortedCitations.filter(matches) : [];
  const citations = citationMatches.slice(0, citationsShown);
  const total = RESEARCH_PAPERS.length + CITING_PAPERS.length;

  const selectView = (next: View) => {
    setView(next);
    setCitationsShown(CITATION_PAGE);
    if (activeTag && ![
      ...(next !== "citations" ? RESEARCH_PAPERS : []),
      ...(next !== "deep" ? CITING_PAPERS : []),
    ].some((p) => p.tags.includes(activeTag))) {
      setActiveTag(null);
    }
  };
  const selectTag = (tag: ResearchTag | null) => {
    setActiveTag(tag);
    setCitationsShown(CITATION_PAGE);
  };

  return (
    <div className="dynamo-blog-home">
      <section
        className="dynamo-blog-latest dynamo-pubs__section"
        id="research"
        aria-labelledby="research-heading"
      >
        <div className="dynamo-blog-section-heading">
          <div className="dynamo-blog-section-heading__copy">
            <span className="dynamo-blog-kicker">From the literature</span>
            <h2 id="research-heading">Research publications</h2>
            <p>
              Deep dives are papers that use, extend, or benchmark Dynamo.
              Citations are papers that cite it in their related work without
              building on it.
            </p>
          </div>
        </div>

        <div className="dynamo-pubs-totals" role="group" aria-label="Choose which papers to show">
          <button type="button" aria-pressed={view === "deep"} onClick={() => selectView("deep")}>
            <strong>{RESEARCH_PAPERS.length}</strong> deep dives
          </button>
          <button
            type="button"
            aria-pressed={view === "citations"}
            onClick={() => selectView("citations")}
          >
            <strong>{CITING_PAPERS.length}</strong> citations
          </button>
          <button type="button" aria-pressed={view === "all"} onClick={() => selectView("all")}>
            <strong>{total}</strong> papers in total
          </button>
        </div>

        <div className="dynamo-pubs-toolbar">
          <div
            className="dynamo-pubs-filter"
            role="group"
            aria-label="Filter papers by keyword"
          >
            <button
              type="button"
              className="dynamo-pubs-chip"
              aria-pressed={activeTag === null}
              onClick={() => selectTag(null)}
            >
              All <span className="dynamo-pubs-chip__count">{inView.length}</span>
            </button>
            {tagCounts.map(({ tag, count }) => (
              <button
                type="button"
                key={tag}
                className="dynamo-pubs-chip"
                aria-pressed={activeTag === tag}
                onClick={() => selectTag(activeTag === tag ? null : tag)}
              >
                {tag} <span className="dynamo-pubs-chip__count">{count}</span>
              </button>
            ))}
          </div>
          <div className="dynamo-pubs-toolbar__meta">
            <span aria-live="polite">
              Showing {deepDives.length + citations.length} of{" "}
              {deepDives.length + citationMatches.length}
            </span>
          </div>
        </div>

        {deepDives.length > 0 ? (
          <div className="dynamo-pubs-group">
            {view === "all" ? <h3 className="dynamo-pubs__group-label">Deep dives</h3> : null}
            <div className="dynamo-pubs">
              {deepDives.map((paper) => (
                <ResearchCard key={paper.url} paper={paper} />
              ))}
            </div>
          </div>
        ) : null}

        {citationMatches.length > 0 ? (
          <div className="dynamo-pubs-group">
            {view === "all" ? (
              <h3
                className={`dynamo-pubs__group-label${deepDives.length > 0 ? " dynamo-pubs__group-label--spaced" : ""}`}
              >
                Citations
              </h3>
            ) : null}
            <div className="dynamo-pubs">
              {citations.map((paper) => (
                <ResearchCard key={paper.url} paper={paper} />
              ))}
            </div>
            {citations.length < citationMatches.length ? (
              <div className="dynamo-pubs-more">
                <button
                  type="button"
                  className="dynamo-blog-button dynamo-blog-button--secondary"
                  onClick={() => setCitationsShown(citationsShown + CITATION_PAGE)}
                >
                  Show {Math.min(CITATION_PAGE, citationMatches.length - citations.length)} more
                </button>
                <button
                  type="button"
                  className="dynamo-blog-button dynamo-blog-button--secondary"
                  onClick={() => setCitationsShown(citationMatches.length)}
                >
                  Show all
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>
    </div>
  );
}

export default ResearchPublications;
