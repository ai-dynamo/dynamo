/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Research publications — papers that use, extend, or benchmark Dynamo.
 *
 * Same card as the partner list next door, with one swap: papers have no
 * publisher logo worth showing, so the venue takes the tile's place. It is
 * also the more useful signal on a paper.
 *
 * Card styles live in PublicationsStyles.tsx, shared with EcosystemPublications;
 * a page rendering this must render <BlogStyles /> and <PublicationsStyles />.
 */
import { RESEARCH_PAPERS, type ResearchPaper } from "./research-papers.data";
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
    </a>
  );
}

// Newest first. Sorted from `iso` rather than trusting the array's order, so a
// new entry can be appended anywhere and still land in the right place.
const sortedPapers = [...RESEARCH_PAPERS].sort((a, b) => b.iso.localeCompare(a.iso));

export function ResearchPublications() {
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
              Papers that use, extend, or benchmark Dynamo, from the teams
              building on it and from the wider systems community.
            </p>
          </div>
        </div>

        <div className="dynamo-pubs">
          {sortedPapers.map((paper) => (
            <ResearchCard key={paper.url} paper={paper} />
          ))}
        </div>
      </section>
    </div>
  );
}

export default ResearchPublications;
