/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * External publications — customer and partner articles about Dynamo.
 *
 * Compact link cards, one per article, in the same two-column rhythm and with
 * the same border, radius and hover treatment as the first-party cards in
 * BlogLanding. They carry less than those do (no summary, no reading time)
 * because we hold no editorial content for them: the card is a signpost to
 * someone else's article, so it shows who published it, when, and the title.
 *
 * Every link leaves the site. Nothing is mirrored or embedded, and that is not
 * only an editorial choice -- most of these publishers send X-Frame-Options or
 * a restrictive frame-ancestors, so an embed would render an empty box for
 * roughly two thirds of the list.
 *
 * Card styles live in PublicationsStyles.tsx, shared with ResearchPublications;
 * a page rendering this must render <BlogStyles /> and <PublicationsStyles />.
 */
"use client";

import { useState } from "react";
import {
  PUBLICATIONS,
  PUBLICATION_TAGS,
  type Publication,
  type PublicationTag,
} from "./publications.data";
import { PUBLISHER_LOGOS } from "./publisher-logos.generated";
import { ExternalMark } from "./PublicationsStyles";

/** Fallback mark: publisher initials, e.g. "Google Cloud" -> "GC". */
function initials(partner: string) {
  return partner
    .replace(/[/&]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase() ?? "")
    .join("");
}

function PublisherMark({ partner }: { partner: string }) {
  const logo = PUBLISHER_LOGOS[partner];
  if (!logo) {
    return (
      <span className="dynamo-pubs__mark" aria-hidden="true">
        {initials(partner)}
      </span>
    );
  }
  return (
    <span className="dynamo-pubs__mark dynamo-pubs__mark--logo" aria-hidden="true">
      <img src={logo} alt="" loading="lazy" decoding="async" />
    </span>
  );
}

function PublicationCard({ publication }: { publication: Publication }) {
  const { title, url, partner, date } = publication;
  // The arrow is tied to the closing word so the two wrap together and it never
  // ends up alone on a line. Split here rather than in a helper component: a
  // helper returning a fragment does not survive Fern's component transform.
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
        <PublisherMark partner={partner} />
        <span className="dynamo-pubs__partner">{partner}</span>
        <span className="dynamo-pubs__date">
          {date}
          {publication.updated ? " · updated" : ""}
        </span>
      </span>
      <span className="dynamo-pubs__title">
        {leadingWords ? `${leadingWords} ` : ""}
        <span className="dynamo-pubs__title-end">
          {lastWord}
          <ExternalMark />
        </span>
      </span>
      <span className="dynamo-pubs__tags">{publication.tags.join(" · ")}</span>
    </a>
  );
}

function siteOf(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

// Title split inline, not in a helper, for the Fern transform reason noted on
// PublicationCard.
function FeaturedCard({ publication }: { publication: Publication }) {
  const { title, url, partner, date, summary } = publication;
  const words = title.trim().split(/\s+/);
  const lastWord = words.pop() ?? "";
  const leadingWords = words.join(" ");
  return (
    <a
      className="dynamo-pubs-featured__card"
      href={url}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="dynamo-pubs-featured__top">
        <PublisherMark partner={partner} />
        <span className="dynamo-pubs-featured__byline">
          <span className="dynamo-pubs__partner">{partner}</span>
          <span className="dynamo-pubs__date">{date}</span>
        </span>
      </span>
      <span className="dynamo-pubs-featured__title">
        {leadingWords ? `${leadingWords} ` : ""}
        <span className="dynamo-pubs__title-end">{lastWord}</span>
      </span>
      {summary ? (
        <span className="dynamo-pubs-featured__summary">{summary}</span>
      ) : null}
      <span className="dynamo-pubs-featured__cta">
        Read on {siteOf(url)}
        <ExternalMark />
      </span>
    </a>
  );
}

// Newest first, from `iso`, so ordering does not depend on how the array is
// maintained by hand. Pinned entries sit in the featured row instead.
const pinnedPublications = PUBLICATIONS.filter((p) => p.pinned != null).sort(
  (a, b) => (a.pinned ?? 0) - (b.pinned ?? 0),
);
// Pinned entries also stay in the list, so it remains a complete archive.
const sortedPublications = [...PUBLICATIONS].sort((a, b) =>
  (b.iso ?? "").localeCompare(a.iso ?? ""),
);

const PAGE_SIZES = [12, 24, 48] as const;

// Only keywords that some entry carries, in vocabulary order, with counts.
const tagCounts = PUBLICATION_TAGS.map((tag) => ({
  tag,
  count: PUBLICATIONS.filter((p) => p.tags.includes(tag)).length,
})).filter(({ count }) => count > 0);

export function EcosystemPublications() {
  const [activeTag, setActiveTag] = useState<PublicationTag | null>(null);
  const [pageSize, setPageSize] = useState<number>(PAGE_SIZES[0]);
  const [shown, setShown] = useState<number>(PAGE_SIZES[0]);

  const filtered = activeTag
    ? sortedPublications.filter((p) => p.tags.includes(activeTag))
    : sortedPublications;
  const visible = filtered.slice(0, shown);

  const selectTag = (tag: PublicationTag | null) => {
    setActiveTag(tag);
    setShown(pageSize);
  };
  const selectPageSize = (size: number) => {
    setPageSize(size);
    setShown(size);
  };

  return (
    <div className="dynamo-blog-home">
      <section
        className="dynamo-blog-latest dynamo-pubs__section"
        id="publications"
        aria-labelledby="publications-heading"
      >
        {/* Same heading structure as BlogLanding's "Latest articles", so the
            pages in this tab open identically. */}
        <div className="dynamo-blog-section-heading">
          <div className="dynamo-blog-section-heading__copy">
            <span className="dynamo-blog-kicker">From the ecosystem</span>
            <h2 id="publications-heading">External publications</h2>
            <p>
              Deep dives, benchmarks, and deployment write-ups about Dynamo,
              published by the customers and partners running it.
            </p>
          </div>
        </div>

        {/* A wrapper rather than a fragment: fragments do not survive Fern's
            component transform. */}
        {pinnedPublications.length > 0 ? (
          <div className="dynamo-pubs-featured-block">
            <h3 className="dynamo-pubs__group-label">Featured</h3>
            <div className="dynamo-pubs-featured">
              {pinnedPublications.map((publication) => (
                <FeaturedCard key={publication.url} publication={publication} />
              ))}
            </div>
            <h3 className="dynamo-pubs__group-label">All publications</h3>
          </div>
        ) : null}

        <div className="dynamo-pubs-toolbar">
          <div
            className="dynamo-pubs-filter"
            role="group"
            aria-label="Filter publications by keyword"
          >
            <button
              type="button"
              className="dynamo-pubs-chip"
              aria-pressed={activeTag === null}
              onClick={() => selectTag(null)}
            >
              All <span className="dynamo-pubs-chip__count">{PUBLICATIONS.length}</span>
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
              Showing {visible.length} of {filtered.length}
            </span>
            <label className="dynamo-pubs-pagesize">
              Per page
              <select
                value={pageSize}
                onChange={(event) => selectPageSize(Number(event.target.value))}
              >
                {PAGE_SIZES.map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
                <option value={PUBLICATIONS.length}>All</option>
              </select>
            </label>
          </div>
        </div>

        <div className="dynamo-pubs">
          {visible.map((publication) => (
            <PublicationCard key={publication.url} publication={publication} />
          ))}
        </div>

        {visible.length < filtered.length ? (
          <div className="dynamo-pubs-more">
            <button
              type="button"
              className="dynamo-blog-button dynamo-blog-button--secondary"
              onClick={() => setShown(shown + pageSize)}
            >
              Show {Math.min(pageSize, filtered.length - visible.length)} more
            </button>
          </div>
        ) : null}
      </section>
    </div>
  );
}

export default EcosystemPublications;
