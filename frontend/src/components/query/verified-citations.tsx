"use client";

import type { Citation } from "@/types";
import { sourceUrl } from "@/lib/source-url";

export function VerifiedCitations({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <details className="mt-3 rounded-lg border bg-muted/20 p-3 text-xs">
      <summary className="cursor-pointer font-medium">
        Original quotations · {citations.length}
      </summary>
      <div className="mt-3 space-y-4">
        {citations.map((citation) => {
          const href = sourceUrl(citation.source_url);
          const level =
            citation.evidence_type === "full_text"
              ? "Full text"
              : citation.evidence_type === "abstract_only"
                ? "Abstract only"
                : "Metadata only";
          return (
            <article key={citation.citation_id} className="space-y-2">
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
                <span className="font-semibold">
                  {citation.source_origin === "web" ? "Web" : "Local"} · {level}
                </span>
                {citation.evidence_type === "full_text" &&
                  citation.pdf_page !== null &&
                  citation.pdf_page > 0 && (
                    <span>Page {citation.pdf_page}</span>
                  )}
              </div>
              <blockquote className="whitespace-pre-wrap border-l-2 border-emerald-600 pl-3 text-sm leading-6">
                {citation.raw_quote}
              </blockquote>
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
                {citation.doi && <span>DOI {citation.doi}</span>}
                <span title={citation.version}>
                  Version {citation.version.slice(0, 20)}
                  {citation.version.length > 20 ? "…" : ""}
                </span>
                {citation.source_origin === "web" && (
                  <span>License: {citation.license || "Unknown"}</span>
                )}
                {citation.retrieved_at && (
                  <time dateTime={citation.retrieved_at}>
                    Retrieved {citation.retrieved_at.slice(0, 10)}
                  </time>
                )}
              </div>
              {href && (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block font-medium text-emerald-700 hover:underline"
                >
                  Open original source
                </a>
              )}
            </article>
          );
        })}
      </div>
    </details>
  );
}
