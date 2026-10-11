"use client";

import { useEffect, useState } from "react";
import { comparePapers, listAllPapers } from "@/lib/api";
import type { CompareResult, PaperResponse } from "@/types";

export default function ComparePage() {
  const [papers, setPapers] = useState<PaperResponse[]>([]);
  const [paperFilter, setPaperFilter] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { listAllPapers().then(setPapers).catch((reason) => setError(String(reason))); }, []);
  async function compare() {
    setLoading(true); setError("");
    try { setResult(await comparePapers(selected, question || undefined)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Comparison failed"); }
    finally { setLoading(false); }
  }
  return <main className="mx-auto max-w-6xl px-4 py-10 md:py-16">
    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700">Cross-paper research</p>
    <h1 className="mt-3 text-4xl font-semibold tracking-tight">Compare evidence</h1>
    <p className="mt-4 max-w-2xl text-muted-foreground">Choose two to five papers. Every comparison cell links back to a local source excerpt.</p>
    <div className="mt-8 grid gap-6 md:grid-cols-[1fr_2fr]">
      <aside className="rounded-xl border bg-card p-5">
        <h2 className="font-semibold">Your library</h2><p className="mb-4 text-sm text-muted-foreground">{selected.length}/5 selected</p>
        <input value={paperFilter} onChange={(event) => setPaperFilter(event.target.value)} placeholder="Find a paper" className="mb-3 w-full rounded-md border bg-background p-2 text-sm" />
        <div className="max-h-96 space-y-2 overflow-y-auto">{papers.filter((paper) => paper.title.toLowerCase().includes(paperFilter.toLowerCase())).slice(0, 100).map((paper) => <label key={paper.id} className="flex cursor-pointer gap-3 rounded-lg border p-3 text-sm hover:bg-muted"><input type="checkbox" checked={selected.includes(paper.id)} disabled={!selected.includes(paper.id) && selected.length >= 5} onChange={() => setSelected((prev) => prev.includes(paper.id) ? prev.filter((id) => id !== paper.id) : [...prev, paper.id])} className="accent-emerald-700" /><span>{paper.title}</span></label>)}</div>
        {papers.length === 0 && <p className="text-sm text-muted-foreground">Import papers in Discover first.</p>}
        <label className="mt-5 block text-sm font-medium">Optional comparison question<input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What differs in their methods?" className="mt-2 w-full rounded-md border bg-background p-2" /></label>
        <button onClick={compare} disabled={selected.length < 2 || loading} className="mt-4 w-full rounded-lg bg-emerald-700 px-4 py-3 text-sm font-semibold text-white disabled:opacity-50">{loading ? "Comparing…" : "Compare selected papers"}</button>
        {error && <p role="alert" className="mt-3 text-sm text-red-600">{error}</p>}
      </aside>
      <section className="min-w-0">{result ? <div className="overflow-x-auto rounded-xl border"><table className="w-full min-w-[680px] border-collapse text-left text-sm"><thead><tr className="bg-muted"><th className="w-40 border-b p-4">Dimension</th>{result.papers.map((paper) => <th className="min-w-56 border-b p-4" key={paper.id}>{paper.title}</th>)}</tr></thead><tbody>{result.rows.map((row) => <tr key={row.label} className="align-top"><th className="border-b bg-muted/50 p-4 font-semibold">{row.label}</th>{row.cells.map((cell) => <td key={cell.paper_id} className="border-b p-4"><p className="whitespace-pre-wrap leading-6">{cell.text}</p>{cell.evidence.map((source) => <a key={source.chunk_id} href={source.source_url ? `${source.source_url}${source.page_start ? `#page=${source.page_start}` : ""}` : "#"} target="_blank" rel="noopener noreferrer" className="mt-3 block rounded-md border bg-muted/40 p-2 text-xs hover:underline"><span className="font-semibold">{source.content_scope === "abstract" ? "Abstract" : source.page_start ? `Page ${source.page_start}` : "Source"}</span> · {source.excerpt.slice(0, 160)}…</a>)}</td>)}</tr>)}</tbody></table></div> : <div className="flex min-h-80 items-center justify-center rounded-xl border border-dashed p-8 text-center text-muted-foreground">Select papers to build an evidence matrix.</div>}</section>
    </div>
  </main>;
}
