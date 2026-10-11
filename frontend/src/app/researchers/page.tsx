"use client";

import { useState } from "react";
import { ArrowUpRight, Loader2, Search, Users } from "lucide-react";
import { getResearcher, searchResearchers } from "@/lib/api";
import type { ResearcherProfile, ResearcherSearchResult } from "@/types/researcher";

export default function ResearchersPage() {
  const [name, setName] = useState("");
  const [institution, setInstitution] = useState("");
  const [discipline, setDiscipline] = useState("");
  const [topic, setTopic] = useState("");
  const [result, setResult] = useState<ResearcherSearchResult | null>(null);
  const [profile, setProfile] = useState<ResearcherProfile | null>(null);
  const [searching, setSearching] = useState(false);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [error, setError] = useState("");

  async function runSearch(event: React.FormEvent) {
    event.preventDefault();
    setSearching(true);
    setError("");
    setProfile(null);
    try {
      setResult(await searchResearchers({ name, institution, discipline, topic }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Researcher search failed");
    } finally {
      setSearching(false);
    }
  }

  async function showProfile(id: string) {
    setLoadingProfile(true);
    setError("");
    try {
      setProfile(await getResearcher(id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load profile");
    } finally {
      setLoadingProfile(false);
    }
  }

  const canSearch = Boolean(name.trim() || institution.trim() || discipline.trim() || topic.trim());

  return <main className="mx-auto max-w-6xl px-4 py-10 md:py-16">
    <header className="mb-9 max-w-3xl">
      <p className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700"><Users className="h-4 w-4" /> Australian researcher directory</p>
      <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">Find people behind the research.</h1>
      <p className="mt-5 leading-7 text-muted-foreground">Search public OpenAlex author profiles whose latest recorded institution is in Australia. Affiliation and author matching may be incomplete; verify details at the source.</p>
    </header>
    <form onSubmit={runSearch} className="rounded-2xl border bg-card p-4 shadow-sm md:p-6">
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="grid gap-1 text-sm font-medium">Researcher name<input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Jane Smith" className="rounded-lg border bg-background px-3 py-2.5 font-normal" /></label>
        <label className="grid gap-1 text-sm font-medium">Australian institution<input value={institution} onChange={(event) => setInstitution(event.target.value)} placeholder="e.g. University of Sydney" className="rounded-lg border bg-background px-3 py-2.5 font-normal" /></label>
        <label className="grid gap-1 text-sm font-medium">Discipline<input value={discipline} onChange={(event) => setDiscipline(event.target.value)} placeholder="e.g. Computer Science" className="rounded-lg border bg-background px-3 py-2.5 font-normal" /></label>
        <label className="grid gap-1 text-sm font-medium">Research topic<input value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="e.g. Machine Learning" className="rounded-lg border bg-background px-3 py-2.5 font-normal" /></label>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">Institution, discipline and topic searches use the closest OpenAlex match. The matched values appear with the results.</p>
      <button disabled={!canSearch || searching} className="mt-5 inline-flex items-center gap-2 rounded-lg bg-emerald-700 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-50">{searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} Search researchers</button>
    </form>
    {error && <p role="alert" className="mt-5 rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">{error}</p>}
    {result && <section className="mt-9">
      <h2 className="text-2xl font-semibold">Researchers</h2>
      <p className="mt-1 text-sm text-muted-foreground">{result.total.toLocaleString()} matching OpenAlex profiles · showing up to {result.researchers.length}</p>
      {Object.keys(result.resolved_filters).length > 0 && <p className="mt-2 text-sm text-muted-foreground">Matched filters: {Object.entries(result.resolved_filters).map(([key, value]) => `${key}: ${value}`).join(" · ")}</p>}
      {result.researchers.length === 0 && <p className="mt-6 rounded-lg border border-dashed p-6 text-sm text-muted-foreground">No matching Australian profiles found. Try a broader name, institution or topic.</p>}
      <div className="mt-5 grid gap-4 md:grid-cols-2">{result.researchers.map((person) => <article key={person.id} className="rounded-xl border bg-card p-5">
        <h3 className="text-lg font-semibold">{person.name}</h3>
        <p className="mt-1 text-sm text-muted-foreground">Latest recorded AU affiliation: {person.institutions.join(", ")}</p>
        {person.fields.length > 0 && <p className="mt-2 text-sm">{person.fields.join(" · ")}</p>}
        {person.topics.length > 0 && <div className="mt-3 flex flex-wrap gap-1">{person.topics.map((value) => <span key={value} className="rounded-full bg-muted px-2 py-1 text-xs">{value}</span>)}</div>}
        <p className="mt-3 text-xs text-muted-foreground">{person.works_count} publications · {person.cited_by_count} citations in OpenAlex</p>
        <div className="mt-4 flex items-center gap-4"><button onClick={() => void showProfile(person.id)} disabled={loadingProfile} className="text-sm font-semibold text-emerald-700 underline disabled:opacity-50">View publications and coauthors</button><a href={person.source_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-sm underline">OpenAlex <ArrowUpRight className="h-3 w-3" /></a></div>
      </article>)}</div>
    </section>}
    {profile && <section className="mt-9 rounded-2xl border bg-card p-5 md:p-7" aria-label="Researcher profile">
      <div className="flex items-start justify-between gap-3"><div><h2 className="text-2xl font-semibold">{profile.name}</h2><p className="mt-1 text-sm text-muted-foreground">Latest recorded AU affiliation: {profile.institutions.join(", ")}</p></div><button onClick={() => setProfile(null)} className="text-sm underline">Close</button></div>
      {profile.orcid && <a href={profile.orcid} target="_blank" rel="noopener noreferrer" className="mt-3 inline-flex items-center gap-1 text-sm underline">ORCID <ArrowUpRight className="h-3 w-3" /></a>}
      <h3 className="mt-7 text-lg font-semibold">Recent publications</h3>
      <p className="text-xs text-muted-foreground">Latest eight works recorded by OpenAlex; citation counts are indicators, not quality ratings.</p>
      <div className="mt-3 space-y-3">{profile.publications.map((work) => <div key={work.id} className="rounded-lg border p-3"><a href={work.source_url} target="_blank" rel="noopener noreferrer" className="font-medium underline">{work.title} <ArrowUpRight className="inline h-3 w-3" /></a><p className="mt-1 text-xs text-muted-foreground">{work.year || "Year unknown"} · {work.citation_count} citations{work.topic ? ` · ${work.topic}` : ""}</p></div>)}</div>
      <h3 className="mt-7 text-lg font-semibold">Coauthors on these publications</h3>
      <div className="mt-3 flex flex-wrap gap-2">{profile.coauthors.length ? profile.coauthors.map((coauthor) => <a key={coauthor.id} href={`https://openalex.org/${coauthor.id}`} target="_blank" rel="noopener noreferrer" className="rounded-full border px-3 py-1 text-sm underline">{coauthor.name}</a>) : <p className="text-sm text-muted-foreground">No coauthors in the displayed publications.</p>}</div>
    </section>}
  </main>;
}
