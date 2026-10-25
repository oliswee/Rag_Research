"use client";

import { Button } from "@/components/ui/button";
import {
  ConversationHistory,
  SearchInput,
  type Message,
} from "@/components/query";
import { listAllPapers, submitQuery } from "@/lib/api";
import type { PaperResponse } from "@/types";
import { AlertCircle, Plus, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

const SESSION_KEY = "foliotrace-session-id";
const SCOPE_KEY = "foliotrace-query-scope-v1";
type Provider = "" | "openalex" | "semantic_scholar" | "pubmed" | "arxiv";
interface SavedScope {
  sessionId: string;
  selectedPapers: string[];
  yearFrom: string;
  yearTo: string;
  localOnly: boolean;
  onlyUploaded: boolean;
  latest: boolean;
  provider: Provider;
}

function readSavedScope(): SavedScope | null {
  if (typeof window === "undefined") return null;
  try {
    const value = JSON.parse(sessionStorage.getItem(SCOPE_KEY) || "null");
    if (
      !value ||
      typeof value.sessionId !== "string" ||
      value.sessionId !== sessionStorage.getItem(SESSION_KEY) ||
      !Array.isArray(value.selectedPapers) ||
      value.selectedPapers.length > 100 ||
      !value.selectedPapers.every(
        (id: unknown) => typeof id === "string" && id.length <= 200
      ) ||
      typeof value.localOnly !== "boolean" ||
      typeof value.onlyUploaded !== "boolean" ||
      typeof value.latest !== "boolean" ||
      !["", "openalex", "semantic_scholar", "pubmed", "arxiv"].includes(
        value.provider
      ) ||
      typeof value.yearFrom !== "string" ||
      typeof value.yearTo !== "string" ||
      !/^\d{0,4}$/.test(value.yearFrom) ||
      !/^\d{0,4}$/.test(value.yearTo)
    )
      return null;
    return value;
  } catch {
    return null;
  }
}

export default function QueryPage() {
  const [savedScope] = useState(readSavedScope);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(
    savedScope?.sessionId || null
  );
  const [papers, setPapers] = useState<PaperResponse[]>([]);
  const [paperFilter, setPaperFilter] = useState("");
  const [selectedPapers, setSelectedPapers] = useState<string[]>(
    savedScope?.selectedPapers || []
  );
  const [yearFrom, setYearFrom] = useState(savedScope?.yearFrom || "");
  const [yearTo, setYearTo] = useState(savedScope?.yearTo || "");
  const [localOnly, setLocalOnly] = useState(savedScope?.localOnly || false);
  const [onlyUploaded, setOnlyUploaded] = useState(
    savedScope?.onlyUploaded || false
  );
  const [latest, setLatest] = useState(savedScope?.latest || false);
  const [provider, setProvider] = useState<Provider>(
    savedScope?.provider || ""
  );

  useEffect(() => {
    listAllPapers()
      .then(setPapers)
      .catch(() => {});
  }, []);

  // Save session ID to sessionStorage
  useEffect(() => {
    try {
      if (sessionId) {
        sessionStorage.setItem(SESSION_KEY, sessionId);
        sessionStorage.setItem(
          SCOPE_KEY,
          JSON.stringify({
            sessionId,
            selectedPapers,
            yearFrom,
            yearTo,
            localOnly,
            onlyUploaded,
            latest,
            provider,
          })
        );
      }
    } catch {
      /* Storage can be unavailable; the current page still keeps scope. */
    }
  }, [
    sessionId,
    selectedPapers,
    yearFrom,
    yearTo,
    localOnly,
    onlyUploaded,
    latest,
    provider,
  ]);

  // Submit a query
  const handleSubmit = useCallback(
    async (query: string) => {
      // Add user message
      setMessages((prev) => [
        ...prev,
        { type: "user", content: query, timestamp: new Date() },
      ]);
      setLoading(true);
      setError(null);

      try {
        const response = await submitQuery(query, sessionId || undefined, {
          retrieval_mode: "hybrid",
          rewrite_query: true,
          paper_ids: selectedPapers.length ? selectedPapers : null,
          year_from: yearFrom ? Number(yearFrom) : null,
          year_to: yearTo ? Number(yearTo) : null,
          provider: provider || null,
          local_only: localOnly,
          only_uploaded: onlyUploaded,
          latest,
          intent: null,
        });

        // Update session ID
        if (response.session_id) {
          setSessionId(response.session_id);
        }
        if (response.routing) {
          setLocalOnly(response.routing.local_only);
          setOnlyUploaded(response.routing.only_uploaded);
        }

        // Add assistant message
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: response.response,
            response,
            timestamp: new Date(),
          },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to get response");
        // Remove the user message on error
        setMessages((prev) => prev.slice(0, -1));
      } finally {
        setLoading(false);
      }
    },
    [
      sessionId,
      selectedPapers,
      yearFrom,
      yearTo,
      provider,
      localOnly,
      onlyUploaded,
      latest,
    ]
  );

  // Handle follow-up suggestion click
  const handleFollowUp = useCallback(
    (suggestion: string) => {
      handleSubmit(suggestion);
    },
    [handleSubmit]
  );

  // Start a new conversation
  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    sessionStorage.removeItem(SESSION_KEY);
    sessionStorage.removeItem(SCOPE_KEY);
    setError(null);
  }, []);

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      {/* Header */}
      <div className="border-b px-4 py-3">
        <div className="mx-auto flex max-w-3xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-foreground">Query</h1>
            <p className="text-sm text-muted-foreground">
              Ask questions about your research library
            </p>
          </div>
          {messages.length > 0 && (
            <Button variant="outline" size="sm" onClick={handleNewConversation}>
              <Plus className="h-4 w-4" />
              New Conversation
            </Button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col overflow-hidden px-4 py-6">
        {/* Error */}
        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            {error}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setError(null)}
              className="ml-auto h-6 w-6 p-0"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        )}

        {/* Conversation */}
        <ConversationHistory
          messages={messages}
          loading={loading}
          onFollowUp={handleFollowUp}
        />

        {/* Input */}
        <div className="mt-4 border-t pt-4">
          <fieldset
            className="mb-3 rounded-lg border bg-card p-3 text-sm"
            disabled={loading}
          >
            <legend className="px-1 font-medium">Evidence access</legend>
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={localOnly}
                  onChange={(event) => setLocalOnly(event.target.checked)}
                />
                Local only
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={onlyUploaded}
                  onChange={(event) => setOnlyUploaded(event.target.checked)}
                />
                Only uploaded papers
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={latest}
                  onChange={(event) => setLatest(event.target.checked)}
                />
                Find latest papers
              </label>
            </div>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">
              {localOnly || onlyUploaded
                ? "Web access is off. Latest results may be incomplete."
                : "Searches your library first. Academic Web evidence may supplement gaps and is not automatically imported."}
            </p>
          </fieldset>
          <details className="mb-4 rounded-lg border bg-card p-3 text-sm">
            <summary className="cursor-pointer font-medium">
              Search scope
            </summary>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <label className="grid gap-1">
                Source
                <select
                  value={provider}
                  onChange={(event) =>
                    setProvider(event.target.value as typeof provider)
                  }
                  className="rounded-md border bg-background p-2"
                >
                  <option value="">All sources</option>
                  <option value="semantic_scholar">Semantic Scholar</option>
                  <option value="openalex">OpenAlex</option>
                  <option value="pubmed">PubMed</option>
                  <option value="arxiv">arXiv</option>
                </select>
              </label>
              <label className="grid gap-1">
                From year
                <input
                  type="number"
                  value={yearFrom}
                  onChange={(event) => setYearFrom(event.target.value)}
                  className="rounded-md border bg-background p-2"
                />
              </label>
              <label className="grid gap-1">
                To year
                <input
                  type="number"
                  value={yearTo}
                  onChange={(event) => setYearTo(event.target.value)}
                  className="rounded-md border bg-background p-2"
                />
              </label>
            </div>
            {papers.length > 0 && (
              <div className="mt-4">
                <p className="mb-2 font-medium">
                  Papers · leave unselected to search the whole library
                </p>
                <input
                  value={paperFilter}
                  onChange={(event) => setPaperFilter(event.target.value)}
                  placeholder="Find a paper"
                  className="mb-2 w-full rounded-md border bg-background p-2"
                />
                <div className="max-h-32 space-y-1 overflow-y-auto">
                  {papers
                    .filter((paper) =>
                      paper.title
                        .toLowerCase()
                        .includes(paperFilter.toLowerCase())
                    )
                    .slice(0, 100)
                    .map((paper) => (
                      <label key={paper.id} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={selectedPapers.includes(paper.id)}
                          onChange={() =>
                            setSelectedPapers((prev) =>
                              prev.includes(paper.id)
                                ? prev.filter((id) => id !== paper.id)
                                : [...prev, paper.id]
                            )
                          }
                        />
                        {paper.title}
                      </label>
                    ))}
                </div>
              </div>
            )}
          </details>
          <SearchInput
            onSubmit={handleSubmit}
            loading={loading}
            placeholder={
              messages.length > 0
                ? "Ask a follow-up question..."
                : "Ask a question about your research..."
            }
          />
        </div>
      </div>
    </div>
  );
}
