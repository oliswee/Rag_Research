"use client";

import { useState } from "react";
import { getPaperContext } from "@/lib/api";
import type { Candidate, PaperContextResult } from "@/types";

export default function PaperContext({ provider, providerId }: {
  provider: Candidate["provider"];
  providerId: string;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [context, setContext] = useState<PaperContextResult | null>(null);
  const [error, setError] = useState("");

  async function toggle() {
    setOpen((previous) => !previous);
    if (context || loading || open) return;
    setLoading(true);
    setError("");
    try {
      setContext(await getPaperContext(provider, providerId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load paper context");
    } finally {
      setLoading(false);
    }
  }

  return <div className="mt-4 border-t pt-3 text-sm">
    <button type="button" onClick={() => void toggle()} aria-expanded={open} className="font-semibold underline underline-offset-2">{open ? "Hide metadata and citations" : "Explore metadata and citations"}</button>
    {open && <div className="mt-3 space-y-3 text-muted-foreground">
      {loading && <p role="status">Loading metadata and citations…</p>}
      {error && <p role="alert" className="text-red-700">{error}</p>}
      {context && <>
        {context.metadata ? <div><p className="font-medium text-foreground">DOI metadata · Crossref</p><dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          {context.metadata.venue && <><dt>Venue</dt><dd>{context.metadata.venue}</dd></>}
          {context.metadata.publisher && <><dt>Publisher</dt><dd>{context.metadata.publisher}</dd></>}
          {context.metadata.type && <><dt>Type</dt><dd>{context.metadata.type}</dd></>}
          {context.metadata.published && <><dt>Published</dt><dd>{context.metadata.published}</dd></>}
          {context.metadata.reference_count != null && <><dt>Deposited references</dt><dd>{context.metadata.reference_count}</dd></>}
        </dl></div> : <p>No Crossref DOI metadata available.</p>}
        <div><p className="font-medium text-foreground">Citing papers · Semantic Scholar</p>{context.citing_papers.length ? <ul className="mt-1 list-disc space-y-1 pl-5">{context.citing_papers.map((paper) => <li key={paper.url}><a href={paper.url} target="_blank" rel="noopener noreferrer" className="underline">{paper.title}</a>{paper.year ? ` · ${paper.year}` : ""}</li>)}</ul> : <p>No citing papers returned.</p>}</div>
        <div><p className="font-medium text-foreground">References · OpenAlex</p>{context.references.length ? <ul className="mt-1 list-disc space-y-1 pl-5">{context.references.map((paper) => <li key={paper.url}><a href={paper.url} target="_blank" rel="noopener noreferrer" className="underline">{paper.title}</a>{paper.year ? ` · ${paper.year}` : ""}</li>)}</ul> : <p>No reference records returned.</p>}</div>
        {context.errors.length > 0 && <p role="status" className="text-amber-700">Some context sources were unavailable: {context.errors.join("; ")}.</p>}
      </>}
    </div>}
  </div>;
}
