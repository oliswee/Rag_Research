"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, BookOpen, Loader2, Search } from "lucide-react";
import { discoverPapers, getImport, retryImport, startImport } from "@/lib/api";
import type { Candidate, DiscoveryResult, ImportJob } from "@/types";

const sourceNames = { openalex: "OpenAlex", semantic_scholar: "Semantic Scholar", pubmed: "PubMed", arxiv: "arXiv" };
const topicKey = (tag: string) => tag.trim().toLocaleLowerCase();

function paperIdentity(paper: Candidate) {
  return paper.doi ? `doi:${paper.doi.toLowerCase()}` : `${paper.provider}:${paper.provider_id}`;
}

function appendUnique(previous: Candidate[], incoming: Candidate[]) {
  const seen = new Set(previous.map(paperIdentity));
  return [...previous, ...incoming.filter((paper) => {
    const identity = paperIdentity(paper);
    if (seen.has(identity)) return false;
    seen.add(identity);
    return true;
  })];
}

export default function DiscoverPage() {
  const [query, setQuery] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [openOnly, setOpenOnly] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [activeSearch, setActiveSearch] = useState<Parameters<typeof discoverPapers>[0] | null>(null);
  const [offsets, setOffsets] = useState<DiscoveryResult["next_offsets"]>({ semantic_scholar: 0, openalex: 0, pubmed: 0, arxiv: 0 });
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [showAllTags, setShowAllTags] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [savedKeys, setSavedKeys] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [searching, setSearching] = useState(false);
  const [saving, setSaving] = useState(false);
  const [job, setJob] = useState<ImportJob | null>(null);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      getImport(job.job_id).then((updated) => {
        setJob(updated);
        setSavedKeys((previous) => Array.from(new Set([
          ...previous,
          ...updated.items.filter((item) => ["done", "duplicate"].includes(item.status))
            .map((item) => `${item.provider}:${item.provider_id}`),
        ])));
      }).catch(() => setError("Could not refresh import progress."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  async function searchPapers(event: React.FormEvent) {
    event.preventDefault();
    if (query.trim().length < 2) return;
    setSearching(true);
    setError("");
    setWarnings([]);
    setSelected([]);
    setSelectedTags([]);
    setHasMore(false);
    try {
      const params = {
        query: query.trim(),
        year_from: yearFrom ? Number(yearFrom) : undefined,
        year_to: yearTo ? Number(yearTo) : undefined,
        open_access: openOnly,
      };
      const result = await discoverPapers(params);
      setCandidates(result.candidates);
      setWarnings(result.errors);
      setActiveSearch(params);
      setOffsets(result.next_offsets);
      setHasMore(result.has_more);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Search failed");
    } finally {
      setSearching(false);
    }
  }

  async function loadMore() {
    if (!activeSearch || !hasMore || loadingMore) return;
    setLoadingMore(true);
    setError("");
    try {
      const result = await discoverPapers({
        ...activeSearch,
        semantic_offset: offsets.semantic_scholar,
        openalex_offset: offsets.openalex,
        pubmed_offset: offsets.pubmed,
        arxiv_offset: offsets.arxiv,
      });
      const combined = appendUnique(candidates, result.candidates);
      setCandidates(combined);
      setOffsets(result.next_offsets);
      setHasMore(result.has_more);
      setWarnings((previous) => [...new Set([...previous, ...result.errors])]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load more papers");
    } finally {
      setLoadingMore(false);
    }
  }

  function toggleTag(tag: string) {
    const key = topicKey(tag);
    setSelected([]);
    setSelectedTags((previous) => previous.includes(key) ? previous.filter((value) => value !== key) : [...previous, key]);
  }

  async function addToLibrary(papers: Candidate[]) {
    setSaving(true);
    setError("");
    try {
      const result = await startImport(papers);
      setJob({ job_id: result.job_id, status: result.status, items: [] });
      setSelected([]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add paper to library");
    } finally {
      setSaving(false);
    }
  }

  const busy = saving || job?.status === "queued" || job?.status === "running";
  const saved = new Set(savedKeys);
  const tagCounts = new Map<string, { label: string; count: number }>();
  candidates.forEach((paper) => paper.tags.forEach((tag) => {
    const key = topicKey(tag);
    const previous = tagCounts.get(key);
    tagCounts.set(key, { label: previous?.label || tag, count: (previous?.count || 0) + 1 });
  }));
  const availableTags = [...tagCounts.values()].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
  const visibleCandidates = selectedTags.length === 0 ? candidates : candidates.filter((paper) => paper.tags.some((tag) => selectedTags.includes(topicKey(tag))));

  return <main className="mx-auto max-w-6xl px-4 py-10 md:py-16">
    <header className="mb-10 max-w-3xl">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700">Research starts here</p>
      <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">Discover the papers.<br /><span className="text-muted-foreground">Keep the evidence.</span></h1>
      <p className="mt-5 leading-7 text-muted-foreground">Search Semantic Scholar, OpenAlex, arXiv, and PubMed. Add a paper to your local library to keep its available full text or abstract.</p>
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
      <p className="mb-5 text-sm text-muted-foreground">Showing {visibleCandidates.length} of {candidates.length} loaded papers · add any paper to your library</p>
      {availableTags.length > 0 && <div className="mb-5 rounded-xl border bg-card p-4"><div className="mb-3 flex items-center justify-between gap-3"><strong className="text-sm">Filter by source topics</strong>{selectedTags.length > 0 && <button onClick={() => setSelectedTags([])} className="text-sm underline">Clear filters</button>}</div><div className="flex flex-wrap gap-2">{availableTags.slice(0, showAllTags ? undefined : 12).map(({ label, count }) => <button key={topicKey(label)} type="button" aria-label={`Filter by ${label}`} aria-pressed={selectedTags.includes(topicKey(label))} onClick={() => toggleTag(label)} className={`rounded-full border px-3 py-1 text-xs ${selectedTags.includes(topicKey(label)) ? "border-emerald-700 bg-emerald-700 text-white" : "hover:border-emerald-700"}`}>{label} · {count}</button>)}{availableTags.length > 12 && <button onClick={() => setShowAllTags((value) => !value)} className="text-xs underline">{showAllTags ? "Show fewer topics" : `Show all ${availableTags.length} topics`}</button>}</div><p className="mt-3 text-xs text-muted-foreground">Selected topics filter the papers loaded so far. Select multiple topics to show papers matching any of them.</p></div>}
      <p className="mb-5 text-xs text-muted-foreground">Recency, citation, and potential foundation labels describe these search results; they do not rate research quality.</p>
      {selected.length > 0 && <button onClick={() => void addToLibrary(candidates.filter((paper) => selected.includes(`${paper.provider}:${paper.provider_id}`)))} disabled={busy} className="mb-5 rounded-lg bg-foreground px-5 py-2.5 text-sm font-semibold text-background disabled:opacity-50">Add {selected.length} selected to library</button>}
      <div className="grid gap-4 md:grid-cols-2">{visibleCandidates.map((paper) => {
        const key = `${paper.provider}:${paper.provider_id}`;
        const paperSaved = saved.has(key);
        return <article key={`${paper.provider}:${paper.provider_id}`} className="rounded-xl border bg-card p-5">
          <div className="mb-3 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground"><span>{sourceNames[paper.provider]} · {paper.year || "Year unknown"}</span><span>{paper.pdf_url ? "Open PDF" : paper.abstract ? "Abstract" : "Metadata only"}</span></div>
          <h3 className="text-lg font-semibold leading-snug">{paper.title}</h3>
          <p className="mt-2 text-sm text-muted-foreground">{paper.authors.slice(0, 3).join(", ")}{paper.authors.length > 3 ? " et al." : ""}</p>
          <p className="mt-3 text-sm"><strong>Why shown:</strong> {paper.relevance_reason}</p>
          {paper.role_labels.length > 0 && <div className="mt-3 flex flex-wrap gap-1" aria-label="Result signals">{paper.role_labels.map((label) => <span key={label} className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs text-emerald-900">{label}</span>)}</div>}
          {paper.tags.length > 0 && <div className="mt-3 flex flex-wrap gap-1" aria-label="Source topics">{paper.tags.map((tag) => <button key={tag} type="button" aria-label={`Filter by ${tag}`} aria-pressed={selectedTags.includes(topicKey(tag))} onClick={() => toggleTag(tag)} className={`rounded-full px-2 py-1 text-xs ${selectedTags.includes(topicKey(tag)) ? "bg-emerald-700 text-white" : "bg-muted hover:bg-emerald-100"}`}>{tag}</button>)}</div>}
          {paper.cited_by_results.length > 0 && <p className="mt-3 text-xs text-muted-foreground">Referenced by {paper.cited_by_results.map((related, index) => <span key={related.provider_id}>{index > 0 ? ", " : ""}<a href={`https://openalex.org/${related.provider_id}`} target="_blank" rel="noopener noreferrer" className="underline">{related.title}</a></span>)} in these results.</p>}
          {paper.abstract_summary ? <p className="mt-4 line-clamp-4 text-sm leading-6 text-muted-foreground"><strong>From the abstract:</strong> {paper.abstract_summary}</p> : <p className="mt-4 text-sm text-muted-foreground">No abstract available; summary not provided.</p>}
          <p className="mt-3 text-xs text-muted-foreground">{paper.citation_count == null ? "Citation count unavailable" : `${paper.citation_count} citations reported by source`} · Citation count is one signal, not a quality rating.</p>
          {paper.possible_duplicate_ids.length > 0 && <p className="mt-3 text-xs text-amber-700">Similar title already in library; adding will check identifiers.</p>}
          <div className="mt-5 flex flex-wrap items-center gap-3"><button onClick={() => void addToLibrary([paper])} disabled={busy || paperSaved} className="inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium disabled:opacity-50"><BookOpen className="h-4 w-4" />{paperSaved ? "In library" : "Add to library"}</button><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={selected.includes(key)} disabled={busy || paperSaved} onChange={() => setSelected((previous) => previous.includes(key) ? previous.filter((value) => value !== key) : [...previous, key])} /> Select for batch</label>{paper.source_url && <a href={paper.source_url} target="_blank" rel="noopener noreferrer" className="ml-auto inline-flex items-center gap-1 text-sm font-medium underline">Source <ArrowUpRight className="h-4 w-4" /></a>}</div>
        </article>;
      })}</div>
      {hasMore && <button type="button" onClick={() => void loadMore()} disabled={loadingMore} className="mx-auto mt-8 flex items-center gap-2 rounded-lg border px-5 py-2.5 text-sm font-semibold hover:bg-muted disabled:opacity-50">{loadingMore && <Loader2 className="h-4 w-4 animate-spin" />} Load more papers</button>}
      <p className="mt-6 text-xs text-muted-foreground">PubMed records are supplied by NCBI. Abstracts may be copyrighted; <a href="https://www.ncbi.nlm.nih.gov/About/disclaimer.html" target="_blank" rel="noopener noreferrer" className="underline">read NCBI&apos;s disclaimer</a>.</p>
    </section>}
  </main>;
}
