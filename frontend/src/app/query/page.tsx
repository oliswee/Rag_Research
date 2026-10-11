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

export default function QueryPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(() =>
    typeof window === "undefined" ? null : sessionStorage.getItem(SESSION_KEY)
  );
  const [papers, setPapers] = useState<PaperResponse[]>([]);
  const [paperFilter, setPaperFilter] = useState("");
  const [selectedPapers, setSelectedPapers] = useState<string[]>([]);
  const [mode, setMode] = useState<"dense" | "bm25" | "hybrid">("dense");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [provider, setProvider] = useState<"" | "openalex" | "semantic_scholar">("");

  useEffect(() => { listAllPapers().then(setPapers).catch(() => {}); }, []);

  // Save session ID to sessionStorage
  useEffect(() => {
    if (sessionId) {
      sessionStorage.setItem(SESSION_KEY, sessionId);
    }
  }, [sessionId]);

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
          retrieval_mode: mode,
          paper_ids: selectedPapers.length ? selectedPapers : undefined,
          year_from: yearFrom ? Number(yearFrom) : undefined,
          year_to: yearTo ? Number(yearTo) : undefined,
          provider: provider || undefined,
        });

        // Update session ID
        if (response.session_id) {
          setSessionId(response.session_id);
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
        setError(
          err instanceof Error ? err.message : "Failed to get response"
        );
        // Remove the user message on error
        setMessages((prev) => prev.slice(0, -1));
      } finally {
        setLoading(false);
      }
    },
    [sessionId, mode, selectedPapers, yearFrom, yearTo, provider]
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
          <details className="mb-4 rounded-lg border bg-card p-3 text-sm"><summary className="cursor-pointer font-medium">Search scope and retrieval</summary><div className="mt-4 grid gap-3 sm:grid-cols-2"><label className="grid gap-1">Mode<select value={mode} onChange={(event) => setMode(event.target.value as typeof mode)} className="rounded-md border bg-background p-2"><option value="dense">Dense (default)</option><option value="bm25">Keyword</option><option value="hybrid">Hybrid · experimental</option></select></label><label className="grid gap-1">Source<select value={provider} onChange={(event) => setProvider(event.target.value as typeof provider)} className="rounded-md border bg-background p-2"><option value="">All sources</option><option value="semantic_scholar">Semantic Scholar</option><option value="openalex">OpenAlex</option></select></label><label className="grid gap-1">From year<input type="number" value={yearFrom} onChange={(event) => setYearFrom(event.target.value)} className="rounded-md border bg-background p-2" /></label><label className="grid gap-1">To year<input type="number" value={yearTo} onChange={(event) => setYearTo(event.target.value)} className="rounded-md border bg-background p-2" /></label></div>{papers.length > 0 && <div className="mt-4"><p className="mb-2 font-medium">Papers · leave unselected to search the whole library</p><input value={paperFilter} onChange={(event) => setPaperFilter(event.target.value)} placeholder="Find a paper" className="mb-2 w-full rounded-md border bg-background p-2" /><div className="max-h-32 space-y-1 overflow-y-auto">{papers.filter((paper) => paper.title.toLowerCase().includes(paperFilter.toLowerCase())).slice(0, 100).map((paper) => <label key={paper.id} className="flex items-center gap-2"><input type="checkbox" checked={selectedPapers.includes(paper.id)} onChange={() => setSelectedPapers((prev) => prev.includes(paper.id) ? prev.filter((id) => id !== paper.id) : [...prev, paper.id])} />{paper.title}</label>)}</div></div>}</details>
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
