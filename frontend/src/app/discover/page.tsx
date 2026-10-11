"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, BookOpen, Loader2, Search } from "lucide-react";
import { discoverPapers, getImport, retryImport, startImport } from "@/lib/api";
import type { Candidate, ImportJob } from "@/types";

export default function DiscoverPage() {
  const [query, setQuery] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [openOnly, setOpenOnly] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [searching, setSearching] = useState(false);
  const [saving, setSaving] = useState(false);
  const [job, setJob] = useState<ImportJob | null>(null);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      getImport(job.job_id).then(setJob).catch(() => setError("Could not refresh import progress."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  async function searchPapers(event: React.FormEvent) {
    event.preventDefault();
    if (query.trim().length < 2) return;
    setSearching(true);
    setError("");
    setWarnings([]);
    try {
      const result = await discoverPapers({
        query: query.trim(),
        year_from: yearFrom ? Number(yearFrom) : undefined,
        year_to: yearTo ? Number(yearTo) : undefined,
        open_access: openOnly,
      });
      setCandidates(result.candidates);
      setWarnings(result.errors);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Search failed");
    } finally {
      setSearching(false);
    }
  }

  async function addToLibrary(paper: Candidate) {
    setSaving(true);
    setError("");
    try {
      const result = await startImport([paper]);
      setJob({ job_id: result.job_id, status: result.status, items: [] });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add paper to library");
    } finally {
      setSaving(false);
    }
  }

  const busy = saving || job?.status === "queued" || job?.status === "running";
  const saved = new Set(job?.items.filter((item) => ["done", "duplicate"].includes(item.status))
    .map((item) => `${item.provider}:${item.provider_id}`) ?? []);

  return <main className="mx-auto max-w-6xl px-4 py-10 md:py-16">
    <header className="mb-10 max-w-3xl">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700">Research starts here</p>
      <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">Discover the papers.<br /><span className="text-muted-foreground">Keep the evidence.</span></h1>
      <p className="mt-5 leading-7 text-muted-foreground">Search Semantic Scholar and OpenAlex. Add a paper to your local library to keep its available full text or abstract.</p>
    </header>
    <form onSubmit={searchPapers} className="rounded-2xl border bg-card p-4 shadow-sm md:p-6">
      <label htmlFor="research-query" className="mb-2 block text-sm font-semibold">Research topic or question</label>
      <div className="flex flex-col gap-3 sm:flex-row">
        <input id="research-query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. How does retrieval improve scientific question answering?" className="min-w-0 flex-1 rounded-lg border bg-background px-4 py-3 text-sm focus:ring-2 focus:ring-emerald-500" />
        <button disabled={searching || query.trim().length < 2} className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 px-6 py-3 text-sm font-semibold text-white disabled:opacity-50">{searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} Search papers</button>
      </div>
      <div className="mt-4 flex flex-wrap items-end gap-4 text-sm">
        <label className="grid gap-1">From year<input type="number" min="1800" max="2100" value={yearFrom} onChange={(event) => setYearFrom(event.target.value)} className="w-28 rounded-md border bg-background px-3 py-2" /></label>
        <label className="grid gap-1">To year<input type="number" min="1800" max="2100" value={yearTo} onChange={(event) => setYearTo(event.target.value)} className="w-28 rounded-md border bg-background px-3 py-2" /></label>
        <label className="flex items-center gap-2 pb-2"><input type="checkbox" checked={openOnly} onChange={(event) => setOpenOnly(event.target.checked)} /> Open access only</label>
      </div>
    </form>
    {error && <p role="alert" className="mt-5 rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">{error}</p>}
    {warnings.length > 0 && <p role="status" className="mt-5 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">Some sources were unavailable. {warnings.join("; ")}. Available results are still shown.</p>}
    {job && <div className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50 p-5 text-sm text-emerald-950 dark:bg-emerald-950 dark:text-emerald-100">
      <strong>Library import: {job.status}</strong>
      <div className="mt-3 space-y-1">{job.items.map((item) => <p key={`${item.provider}:${item.provider_id}`}>{item.provider_id}: {item.status}{item.content_scope ? ` · ${item.content_scope.replace("_", " ")}` : ""}{item.warning ? ` · ${item.warning}` : ""}{item.error ? ` · ${item.error}` : ""}</p>)}</div>
      {job.status === "partial" && <button onClick={() => retryImport(job.job_id).then(() => setJob({ ...job, status: "queued" })).catch((reason) => setError(String(reason)))} className="mt-3 mr-4 font-semibold underline">Retry import</button>}
      {!["queued", "running"].includes(job.status) && <Link href="/library" className="mt-3 inline-flex font-semibold underline">Open library <ArrowUpRight className="h-4 w-4" /></Link>}
    </div>}
    {candidates.length > 0 && <section className="mt-10">
      <h2 className="text-2xl font-semibold">Search results</h2>
      <p className="mb-5 text-sm text-muted-foreground">{candidates.length} papers · add any paper to your library</p>
      <div className="grid gap-4 md:grid-cols-2">{candidates.map((paper) => {
        const paperSaved = saved.has(`${paper.provider}:${paper.provider_id}`);
        return <article key={`${paper.provider}:${paper.provider_id}`} className="rounded-xl border bg-card p-5">
          <div className="mb-3 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground"><span>{paper.provider === "openalex" ? "OpenAlex" : "Semantic Scholar"} · {paper.year || "Year unknown"}</span><span>{paper.pdf_url ? "Open PDF" : paper.abstract ? "Abstract" : "Metadata only"}</span></div>
          <h3 className="text-lg font-semibold leading-snug">{paper.title}</h3>
          <p className="mt-2 text-sm text-muted-foreground">{paper.authors.slice(0, 3).join(", ")}{paper.authors.length > 3 ? " et al." : ""}</p>
          <p className="mt-3 text-sm"><strong>Why shown:</strong> {paper.relevance_reason}</p>
          {paper.tags.length > 0 && <div className="mt-3 flex flex-wrap gap-1" aria-label="Source topics">{paper.tags.map((tag) => <span key={tag} className="rounded-full bg-muted px-2 py-1 text-xs">{tag}</span>)}</div>}
          {paper.abstract_summary ? <p className="mt-4 line-clamp-4 text-sm leading-6 text-muted-foreground"><strong>From the abstract:</strong> {paper.abstract_summary}</p> : <p className="mt-4 text-sm text-muted-foreground">No abstract available; summary not provided.</p>}
          <p className="mt-3 text-xs text-muted-foreground">{paper.citation_count == null ? "Citation count unavailable" : `${paper.citation_count} citations reported by source`} · Citation count is one signal, not a quality rating.</p>
          {paper.possible_duplicate_ids.length > 0 && <p className="mt-3 text-xs text-amber-700">Similar title already in library; adding will check identifiers.</p>}
          <div className="mt-5 flex items-center justify-between gap-2"><button onClick={() => void addToLibrary(paper)} disabled={busy || paperSaved} className="inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium disabled:opacity-50"><BookOpen className="h-4 w-4" />{paperSaved ? "In library" : "Add to library"}</button>{paper.source_url && <a href={paper.source_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-sm font-medium underline">Source <ArrowUpRight className="h-4 w-4" /></a>}</div>
        </article>;
      })}</div>
    </section>}
  </main>;
}
