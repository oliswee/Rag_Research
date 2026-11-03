"use client";

import { useEffect, useRef, useState } from "react";
import { getPublicationStatus, refreshPublicationStatus } from "@/lib/api";
import { publicationLabel } from "@/lib/publication";
import { sourceUrl } from "@/lib/source-url";
import type { PublicationRecord } from "@/types";

export function PublicationStatusPanel({ paperId }: { paperId: string }) {
  return <StatusDetails key={paperId} paperId={paperId} />;
}

function StatusDetails({ paperId }: { paperId: string }) {
  const [open, setOpen] = useState(false);
  const [record, setRecord] = useState<PublicationRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => pending.current?.abort(), []);

  async function load(refresh = false) {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    setError("");
    try {
      const response = await (
        refresh ? refreshPublicationStatus : getPublicationStatus
      )(paperId, controller.signal);
      if (!controller.signal.aborted) setRecord(response);
    } catch {
      if (!controller.signal.aborted)
        setError(
          "Status check unavailable. Any previous record is shown below."
        );
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  return (
    <section className="mt-4 border-t pt-3 text-xs text-muted-foreground">
      <button
        type="button"
        className="font-medium text-foreground hover:underline"
        aria-expanded={open}
        onClick={() => {
          setOpen(!open);
          if (!open && !record) void load();
        }}
      >
        Publication status
      </button>
      {open && (
        <div className="mt-3 space-y-3">
          {loading && <p role="status">Checking status…</p>}
          {error && <p role="alert">{error}</p>}
          {record && (
            <>
              <p
                className={
                  record.status === "unknown"
                    ? ""
                    : "font-medium text-amber-700"
                }
              >
                {publicationLabel(record.status)}
              </p>
              <p>Unknown does not mean the paper has no retraction notice.</p>
              {record.doi && <p className="break-all">DOI {record.doi}</p>}
              {record.source_date && <p>Published {record.source_date}</p>}
              {record.source_version && (
                <p title={record.source_version}>
                  Local source version {record.source_version.slice(0, 24)}…
                </p>
              )}
              {record.checked_at && (
                <p>Last checked {record.checked_at.slice(0, 10)}</p>
              )}
              {(record.stale || record.refresh_error) && (
                <p role="alert">
                  Refresh unavailable; this record may be stale.
                </p>
              )}
              {sourceUrl(record.source_url) && (
                <a
                  className="inline-block text-emerald-700 hover:underline"
                  href={sourceUrl(record.source_url)!}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Public status source
                </a>
              )}
              {!!record.relations.length && (
                <ul className="space-y-2">
                  {record.relations.map((relation, i) => (
                    <li key={i}>
                      {relation.kind.replaceAll("-", " ")} ·{" "}
                      {sourceUrl(relation.url) ? (
                        <a
                          className="break-all text-emerald-700 hover:underline"
                          href={sourceUrl(relation.url)!}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {relation.doi}
                        </a>
                      ) : (
                        relation.doi
                      )}
                    </li>
                  ))}
                </ul>
              )}
              {!!record.updates.length && (
                <ul className="space-y-2">
                  {record.updates.map((event, i) => (
                    <li key={i}>
                      {event.kind.replaceAll("_", " ")} · {event.date} ·{" "}
                      {sourceUrl(event.url) ? (
                        <a
                          className="break-all text-emerald-700 hover:underline"
                          href={sourceUrl(event.url)!}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {event.notice_doi}
                        </a>
                      ) : (
                        event.notice_doi
                      )}
                    </li>
                  ))}
                </ul>
              )}
              {record.notices.includes("conflicting_status_events") && (
                <p>Conflicting publication notices need review.</p>
              )}
              <button
                type="button"
                disabled={loading || !record.doi}
                onClick={() => void load(true)}
                className="rounded-md border px-3 py-2 text-foreground disabled:opacity-50"
              >
                Check public DOI metadata
              </button>
              <p>
                Checks public publication records on request. Only the DOI is
                sent.
              </p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
