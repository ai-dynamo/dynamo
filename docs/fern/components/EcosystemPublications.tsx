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
import { PUBLICATIONS, type Publication } from "./publications.data";
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
    </a>
  );
}

// Newest first, from `iso`, so ordering does not depend on how the array is
// maintained by hand.
const sortedPublications = [...PUBLICATIONS].sort((a, b) =>
  (b.iso ?? "").localeCompare(a.iso ?? ""),
);

export function EcosystemPublications() {
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

        <div className="dynamo-pubs">
          {sortedPublications.map((publication) => (
            <PublicationCard key={publication.url} publication={publication} />
          ))}
        </div>
      </section>
    </div>
  );
}

export default EcosystemPublications;
