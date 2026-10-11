"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, BookOpen, Check, Loader2, Search, Sparkles } from "lucide-react";
import { discoverPapers, getImport, retryImport, startImport } from "@/lib/api";
import type { Candidate, ImportJob } from "@/types";

export default function DiscoverPage() {
  const [query, setQuery] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [openOnly, setOpenOnly] = useState(false);
  const [provider, setProvider] = useState("");
  const [tag, setTag] = useState("");
  const [nextOffset, setNextOffset] = useState<number | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [searching, setSearching] = useState(false);
  const [importing, setImporting] = useState(false);
  const [job, setJob] = useState<ImportJob | null>(null);
  const [estimatedCalls, setEstimatedCalls] = useState(0);
  const [error, setError] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);

  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    const timer = window.setInterval(() => { getImport(job.job_id).then(setJob).catch(() => {}); }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  async function searchPapers(event: React.FormEvent) {
    event.preventDefault();
    if (query.trim().length < 2) return;
    setSearching(true); setError(""); setSelected([]); setJob(null);
    try {
      const result = await discoverPapers({ query: query.trim(), year_from: yearFrom ? Number(yearFrom) : undefined, year_to: yearTo ? Number(yearTo) : undefined, open_access: openOnly, provider: provider || undefined });
      setCandidates(result.candidates); setWarnings(result.errors); setNextOffset(result.next_offset); setTag("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Search failed"); }
    finally { setSearching(false); }
  }

  async function loadMore() {
    if (nextOffset === null) return;
    setLoadingMore(true);
    try {
      const result = await discoverPapers({ query: query.trim(), year_from: yearFrom ? Number(yearFrom) : undefined, year_to: yearTo ? Number(yearTo) : undefined, open_access: openOnly, provider: provider || undefined, offset: nextOffset });
      setCandidates((prev) => {
        const seen = new Set(prev.map((p) => p.doi || p.title.trim().toLowerCase()));
        return [...prev, ...result.candidates.filter((p) => {
          const key = p.doi || p.title.trim().toLowerCase();
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        })];
      });
      setWarnings(result.errors);
      setNextOffset(result.next_offset);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not load more"); }
    finally { setLoadingMore(false); }
  }

  async function importSelected() {
    setImporting(true); setError("");
    try {
      const picked = candidates.filter((paper) => selected.includes(`${paper.provider}:${paper.provider_id}`));
      const result = await startImport(picked);
      setEstimatedCalls(result.estimated_api_calls);
      setJob({ job_id: result.job_id, status: result.status, items: [] });
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Import failed"); }
    finally { setImporting(false); }
  }

  return <main className="mx-auto max-w-6xl px-4 py-10 md:py-16">
    <div className="mb-10 max-w-3xl">
      <p className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700 dark:text-emerald-400"><Sparkles className="h-4 w-4" /> Research starts here</p>
      <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">Discover the papers.<br /><span className="text-muted-foreground">Keep the evidence.</span></h1>
      <p className="mt-5 max-w-2xl text-base leading-7 text-muted-foreground">Search PubMed, Semantic Scholar and OpenAlex, then import open full text or an abstract into your local library. The library stays on this computer.</p>
    </div>
    <form onSubmit={searchPapers} className="rounded-2xl border bg-card p-4 shadow-sm md:p-6">
      <label htmlFor="research-query" className="mb-2 block text-sm font-semibold">Research topic or question</label>
      <div className="flex flex-col gap-3 sm:flex-row">
        <input id="research-query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. How does retrieval improve scientific question answering?" className="min-w-0 flex-1 rounded-lg border bg-background px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-emerald-500" />
        <button disabled={searching || query.trim().length < 2} className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 px-6 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:opacity-50">{searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} Search papers</button>
      </div>
      <div className="mt-4 flex flex-wrap items-end gap-4 text-sm"><label className="grid gap-1 text-muted-foreground">Source<select value={provider} onChange={(e) => setProvider(e.target.value)} className="rounded-md border bg-background px-3 py-2 text-foreground"><option value="">All sources</option><option value="pubmed">PubMed</option><option value="openalex">OpenAlex</option><option value="semantic_scholar">Semantic Scholar</option></select></label>
        <label className="grid gap-1 text-muted-foreground">From year<input type="number" min="1800" max="2100" value={yearFrom} onChange={(e) => setYearFrom(e.target.value)} className="w-28 rounded-md border bg-background px-3 py-2 text-foreground" /></label>
        <label className="grid gap-1 text-muted-foreground">To year<input type="number" min="1800" max="2100" value={yearTo} onChange={(e) => setYearTo(e.target.value)} className="w-28 rounded-md border bg-background px-3 py-2 text-foreground" /></label>
        <label className="flex items-center gap-2 pb-2"><input type="checkbox" checked={openOnly} onChange={(e) => setOpenOnly(e.target.checked)} className="accent-emerald-700" /> Open access only</label>
      </div>
    </form>
    {error && <p role="alert" className="mt-5 rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">{error}</p>}
    {warnings.length > 0 && <p className="mt-5 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">Some sources were unavailable: {warnings.join("; ")}</p>}
    {job && <div className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50 p-5 text-sm text-emerald-950 dark:bg-emerald-950 dark:text-emerald-100"><strong>Import {job.status}</strong><span className="ml-2 text-muted-foreground">{job.items.filter((item) => item.status === "done" || item.status === "duplicate").length}/{job.items.length} processed</span><p className="mt-1 text-xs text-muted-foreground">At least {estimatedCalls} academic API calls estimated; PDF and optional AI processing may add more.</p><div className="mt-3 space-y-1">{job.items.map((item) => <p key={`${item.provider}:${item.provider_id}`}>{item.provider_id}: {item.status}{item.content_scope ? ` · ${item.content_scope.replace("_", " ")}` : ""}{item.warning ? ` · ${item.warning}` : ""}{item.error ? ` · ${item.error}` : ""}</p>)}</div>{job.status === "partial" && <button onClick={() => retryImport(job.job_id).then(() => setJob({ ...job, status: "queued" })).catch((reason) => setError(String(reason)))} className="mt-3 mr-4 font-semibold underline">Retry failed imports</button>}{job.status !== "queued" && job.status !== "running" && <Link href="/library" className="mt-3 inline-flex font-semibold underline">Open library <ArrowUpRight className="h-4 w-4" /></Link>}</div>}
    {candidates.length > 0 && <section className="mt-10"><div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-2xl font-semibold">Search results</h2><p className="text-sm text-muted-foreground">{candidates.length} papers loaded · select the ones you want to keep</p></div><button disabled={!selected.length || importing} onClick={importSelected} className="inline-flex items-center gap-2 rounded-lg bg-foreground px-5 py-2.5 text-sm font-semibold text-background disabled:opacity-50">{importing && <Loader2 className="h-4 w-4 animate-spin" />}Import {selected.length} selected</button></div><div className="mb-4 flex flex-wrap gap-3 text-sm"><label className="grid gap-1">Publication type<select value={tag} onChange={(event) => setTag(event.target.value)} className="rounded-md border bg-background px-3 py-2"><option value="">All types</option>{Array.from(new Set(candidates.flatMap((p) => p.tags || []))).sort().map((value) => <option key={value} value={value}>{value}</option>)}</select></label><button type="button" onClick={() => setSelected(candidates.filter((p) => !tag || p.tags?.includes(tag)).map((p) => `${p.provider}:${p.provider_id}`))} className="self-end rounded-md border px-3 py-2">Select visible</button><button type="button" onClick={() => setSelected([])} className="self-end rounded-md border px-3 py-2">Clear selection</button></div><div className="grid gap-4 md:grid-cols-2">{candidates.filter((paper) => !tag || paper.tags?.includes(tag)).map((paper) => { const key = `${paper.provider}:${paper.provider_id}`; const active = selected.includes(key); return <article key={key} className={`rounded-xl border bg-card p-5 transition ${active ? "border-emerald-600 ring-1 ring-emerald-600" : "hover:border-foreground/30"}`}><div className="mb-3 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground"><span>{paper.provider === "openalex" ? "OpenAlex" : paper.provider === "pubmed" ? "PubMed" : "Semantic Scholar"} · {paper.year || "Year unknown"}</span><span className={paper.pdf_url ? "text-emerald-700" : "text-amber-700"}>{paper.pdf_url ? "Open PDF" : paper.abstract ? "Abstract" : "Metadata only"}</span></div><h3 className="text-lg font-semibold leading-snug">{paper.title}</h3>{Boolean(paper.tags?.length) && <p className="mt-2 text-xs text-muted-foreground">{paper.tags?.slice(0, 3).join(" · ")}</p>}<p className="mt-2 text-sm text-muted-foreground">{paper.authors.slice(0, 3).join(", ")}{paper.authors.length > 3 ? " et al." : ""}</p>{paper.abstract && <p className="mt-4 line-clamp-3 text-sm leading-6 text-muted-foreground">{paper.abstract}</p>}{paper.possible_duplicate_ids.length > 0 && <p className="mt-3 text-xs text-amber-700">Similar title already in library — review before importing.</p>}<div className="mt-5 flex items-center justify-between gap-2"><button onClick={() => setSelected((prev) => active ? prev.filter((value) => value !== key) : [...prev, key])} aria-pressed={active} className={`inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium ${active ? "bg-emerald-700 text-white" : "border hover:bg-muted"}`}>{active ? <Check className="h-4 w-4" /> : <BookOpen className="h-4 w-4" />}{active ? "Selected" : "Select paper"}</button>{paper.source_url && <a href={paper.source_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-sm font-medium underline">Source <ArrowUpRight className="h-4 w-4" /></a>}</div></article>; })}</div>{nextOffset !== null && <div className="mt-6 flex justify-center"><button type="button" disabled={loadingMore} onClick={loadMore} className="rounded-lg border px-6 py-3 text-sm font-semibold disabled:opacity-50">{loadingMore ? "Loading…" : "Load more papers"}</button></div>}</section>}
  </main>;
}
