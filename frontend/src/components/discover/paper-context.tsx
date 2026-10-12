"use client";

import { useState } from "react";
import { getPaperContext } from "@/lib/api";
import type { PaperContextResult } from "@/types";

export default function PaperContext({ doi }: { doi: string }) {
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
      setContext(await getPaperContext(doi));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load paper context");
    } finally {
      setLoading(false);
    }
  }

  return <div className="mt-4 border-t pt-3 text-sm">
    <button type="button" onClick={() => void toggle()} aria-expanded={open} className="font-semibold underline underline-offset-2">{open ? "Hide DOI metadata" : "Explore DOI metadata"}</button>
    {open && <div className="mt-3 space-y-3 text-muted-foreground">
      {loading && <p role="status">Loading DOI metadata…</p>}
      {error && <p role="alert" className="text-red-700">{error}</p>}
      {context && <>
        {context.metadata ? <div><p className="font-medium text-foreground">DOI metadata · Crossref</p><dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          {context.metadata.venue && <><dt>Venue</dt><dd>{context.metadata.venue}</dd></>}
          {context.metadata.publisher && <><dt>Publisher</dt><dd>{context.metadata.publisher}</dd></>}
          {context.metadata.type && <><dt>Type</dt><dd>{context.metadata.type}</dd></>}
          {context.metadata.published && <><dt>Published</dt><dd>{context.metadata.published}</dd></>}
          {context.metadata.reference_count != null && <><dt>Deposited references</dt><dd>{context.metadata.reference_count}</dd></>}
        </dl></div> : <p>No Crossref DOI metadata available.</p>}
        {context.errors.length > 0 && <p role="status" className="text-amber-700">Crossref metadata unavailable: {context.errors.join("; ")}.</p>}
      </>}
    </div>}
  </div>;
}
