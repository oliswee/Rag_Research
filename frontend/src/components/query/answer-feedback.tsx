"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { getFeedbackRun, submitFeedback } from "@/lib/api";
import type { FeedbackManifest, FeedbackReason } from "@/types";

const reasons: [FeedbackReason, string][] = [
  ["wrong_paper", "Wrong paper or passage"],
  ["missed_evidence", "Missed evidence"],
  ["unsupported_citation", "Unsupported citation"],
  ["incorrect_interpretation", "Incorrect interpretation"],
  ["outdated_conflicting", "Outdated or conflicting evidence"],
  ["web_result_irrelevant", "Irrelevant Web result"],
];

export function AnswerFeedback({ runId }: { runId: string }) {
  return <FeedbackForm key={runId} runId={runId} />;
}

function FeedbackForm({ runId }: { runId: string }) {
  const id = useId();
  const controller = useRef<AbortController | null>(null);
  const [open, setOpen] = useState(false);
  const [manifest, setManifest] = useState<FeedbackManifest | null>(null);
  const [reason, setReason] = useState<FeedbackReason>("wrong_paper");
  const [sourceId, setSourceId] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  useEffect(() => () => controller.current?.abort(), []);

  async function expand() {
    setOpen(true);
    setError("");
    setPending(true);
    const request = new AbortController();
    controller.current?.abort();
    controller.current = request;
    try {
      const value = await getFeedbackRun(runId, request.signal);
      if (!request.signal.aborted) setManifest(value);
    } catch {
      if (!request.signal.aborted) setError("Feedback is unavailable for this answer.");
    } finally {
      if (!request.signal.aborted) setPending(false);
    }
  }

  async function save() {
    if (!manifest || pending) return;
    setPending(true);
    setError("");
    const request = new AbortController();
    controller.current?.abort();
    controller.current = request;
    try {
      await submitFeedback({run_id: runId, reason, source_id: sourceId || null}, request.signal);
      if (!request.signal.aborted) setSaved(true);
    } catch {
      if (!request.signal.aborted) setError("Feedback could not be saved. Please retry.");
    } finally {
      if (!request.signal.aborted) setPending(false);
    }
  }

  return (
    <div className="border-t pt-3 text-sm">
      {!open && <Button variant="ghost" size="sm" onClick={expand}>Report an issue</Button>}
      {error && <p role="alert" className="text-destructive">{error}</p>}
      {open && pending && !manifest && <p role="status">Loading feedback options…</p>}
      {saved ? <p role="status" className="text-muted-foreground">Saved locally for review.</p> : open && manifest && (
        <form className="grid gap-3 sm:grid-cols-2" onSubmit={(event) => {event.preventDefault(); void save();}}>
          <label htmlFor={`${id}-reason`} className="grid gap-1 text-muted-foreground">Error type
            <select id={`${id}-reason`} value={reason} disabled={pending} onChange={(event) => setReason(event.target.value as FeedbackReason)} className="rounded-md border bg-background p-2 text-foreground">
              {reasons.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <label htmlFor={`${id}-source`} className="grid gap-1 text-muted-foreground">Evidence
            <select id={`${id}-source`} value={sourceId} disabled={pending} onChange={(event) => setSourceId(event.target.value)} className="rounded-md border bg-background p-2 text-foreground">
              <option value="">Whole answer</option>
              {manifest.sources.map((source, index) => <option key={source.source_id} value={source.source_id}>{source.source_origin === "web" ? "Web" : "Local"} {index + 1} · {source.evidence_type.replaceAll("_", " ")}{source.doi ? ` · ${source.doi}` : ""}</option>)}
            </select>
          </label>
          <div className="sm:col-span-2"><Button type="submit" variant="outline" size="sm" disabled={pending}>{pending ? "Saving…" : "Save feedback"}</Button></div>
        </form>
      )}
    </div>
  );
}
