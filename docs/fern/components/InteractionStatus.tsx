/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * InteractionStatus — one status mark in the pairwise Feature Interactions
 * tables on the Compatibility reference page. Carries the accessible label
 * for the status so the tables read as text, not as bare glyphs, and adds an
 * information marker when the cell has a note for a wrapping <Tooltip>.
 *
 * Server component (no "use client"); the .dynref-interaction-* rules that
 * tint the enclosing cell live in ReferenceStyles.tsx.
 */

type InteractionStatusKind = "yes" | "wip" | "no" | "na";

interface InteractionStatusProps {
  status: InteractionStatusKind;
  noteLabel?: string;
}

const STATUS_LABEL: Record<InteractionStatusKind, string> = {
  yes: "Supported",
  wip: "Work in progress",
  no: "Not supported",
  na: "Not applicable",
};

const STATUS_MARK: Record<InteractionStatusKind, string> = {
  yes: "✓",
  wip: "WIP",
  no: "×",
  na: "—",
};

export function InteractionStatus({ status, noteLabel }: InteractionStatusProps) {
  const label = noteLabel ? `${STATUS_LABEL[status]} — ${noteLabel}` : STATUS_LABEL[status];

  return (
    <span
      className={`dynref-interaction dynref-interaction--${status}`}
      role="img"
      aria-label={label}
      tabIndex={noteLabel ? 0 : undefined}
    >
      <span aria-hidden="true">{STATUS_MARK[status]}</span>
      {noteLabel && (
        <span className="dynref-interaction-note-mark" aria-hidden="true">
          i
        </span>
      )}
    </span>
  );
}
