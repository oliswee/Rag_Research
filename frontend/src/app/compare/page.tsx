"use client";

import { useEffect, useState } from "react";
import { comparePapers, listAllPapers } from "@/lib/api";
import { ComparisonMatrix } from "@/components/research/comparison-matrix";
import type { CompareResult, PaperResponse } from "@/types";

export default function ComparePage() {
  const [papers, setPapers] = useState<PaperResponse[]>([]);
  const [paperFilter, setPaperFilter] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [localOnly, setLocalOnly] = useState(false);
  const [onlyUploaded, setOnlyUploaded] = useState(false);
  const [latest, setLatest] = useState(false);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    listAllPapers()
      .then(setPapers)
      .catch(() => setError("The library could not be loaded."));
  }, []);

  async function compare() {
    setLoading(true);
    setError("");
    try {
      setResult(
        await comparePapers(selected, question || undefined, {
          local_only: localOnly,
          only_uploaded: onlyUploaded,
          latest,
        })
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-7xl px-4 py-10 md:py-14">
      <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700">
        Cross-paper research
      </p>
      <h1 className="mt-3 text-4xl font-semibold tracking-tight">
        Compare evidence
      </h1>
      <p className="mt-4 max-w-2xl leading-7 text-muted-foreground">
        Choose two to five papers. Findings must cite their original evidence;
        unsupported comparisons remain unknown.
      </p>
      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(250px,300px)_minmax(0,1fr)]">
        <aside className="self-start rounded-xl border bg-card p-5">
          <h2 className="font-semibold">Your library</h2>
          <p className="mb-4 mt-1 text-sm text-muted-foreground">
            {selected.length}/5 selected
          </p>
          <fieldset disabled={loading}>
            <label className="block text-sm">
              Find a paper
              <input
                value={paperFilter}
                onChange={(event) => setPaperFilter(event.target.value)}
                className="mb-3 mt-1 w-full rounded-md border bg-background p-2 text-sm"
              />
            </label>
            <div className="max-h-72 space-y-2 overflow-y-auto">
              {papers
                .filter((paper) =>
                  paper.title.toLowerCase().includes(paperFilter.toLowerCase())
                )
                .slice(0, 100)
                .map((paper) => (
                  <label
                    key={paper.id}
                    className="flex cursor-pointer gap-3 rounded-lg border p-3 text-sm hover:bg-muted"
                  >
                    <input
                      type="checkbox"
                      checked={selected.includes(paper.id)}
                      disabled={
                        !selected.includes(paper.id) && selected.length >= 5
                      }
                      onChange={() => {
                        setResult(null);
                        setSelected((prev) =>
                          prev.includes(paper.id)
                            ? prev.filter((id) => id !== paper.id)
                            : [...prev, paper.id]
                        );
                      }}
                      className="mt-1 accent-emerald-700"
                    />
                    <span>
                      {paper.title}
                      <span className="mt-1 block text-xs text-muted-foreground">
                        {paper.import_source ? "Imported" : "Uploaded"}
                        {paper.published_date
                          ? ` · ${paper.published_date.slice(0, 4)}`
                          : ""}{" "}
                        · {paper.id.slice(0, 8)}
                        {onlyUploaded && paper.import_source
                          ? " · excluded by uploaded-only scope"
                          : ""}
                      </span>
                    </span>
                  </label>
                ))}
            </div>
            {papers.length === 0 && (
              <p className="text-sm text-muted-foreground">
                Find or upload papers to select study identities.
              </p>
            )}
            <div className="mt-5 space-y-2 rounded-lg bg-muted/40 p-3 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={localOnly}
                  onChange={(event) => {
                    setLocalOnly(event.target.checked);
                    setResult(null);
                  }}
                />
                Local only
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={onlyUploaded}
                  onChange={(event) => {
                    setOnlyUploaded(event.target.checked);
                    setResult(null);
                  }}
                />
                Only uploaded papers
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={latest}
                  onChange={(event) => {
                    setLatest(event.target.checked);
                    setResult(null);
                  }}
                />
                Find latest papers
              </label>
              <p className="pt-1 text-xs leading-5 text-muted-foreground">
                {localOnly || onlyUploaded
                  ? "Web access is off. Latest coverage may be incomplete."
                  : "Each paper is checked locally first. Permitted Web evidence is temporary and is not automatically imported."}
              </p>
            </div>
            <label className="mt-5 block text-sm font-medium">
              Optional comparison question
              <textarea
                value={question}
                onChange={(event) => {
                  setQuestion(event.target.value);
                  setResult(null);
                }}
                placeholder="What differs in their methods?"
                rows={3}
                className="mt-2 w-full rounded-md border bg-background p-2 text-sm"
              />
            </label>
            <button
              onClick={compare}
              disabled={selected.length < 2 || loading}
              className="mt-4 w-full rounded-lg bg-emerald-700 px-4 py-3 text-sm font-semibold text-white disabled:opacity-50"
            >
              {loading ? "Checking evidence…" : "Compare selected papers"}
            </button>
          </fieldset>
          {error && (
            <p role="alert" className="mt-3 text-sm text-red-600">
              {error}
            </p>
          )}
        </aside>
        <section className="min-w-0" aria-live="polite">
          {result ? (
            <ComparisonMatrix result={result} />
          ) : (
            <div className="flex min-h-80 items-center justify-center rounded-xl border border-dashed p-8 text-center text-muted-foreground">
              {loading
                ? "Retrieving and verifying evidence before publishing findings…"
                : "Select papers to build a verified evidence matrix."}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
