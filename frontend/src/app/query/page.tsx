"use client";

import { Button } from "@/components/ui/button";
import {
  ConversationHistory,
  SearchInput,
  type Message,
} from "@/components/query";
import { submitQuery } from "@/lib/api";
import { AlertCircle, Plus, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

const SESSION_KEY = "research-vault-session-id";

export default function QueryPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Load session ID from sessionStorage
  useEffect(() => {
    const stored = sessionStorage.getItem(SESSION_KEY);
    if (stored) {
      setSessionId(stored);
    }
  }, []);

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
        const response = await submitQuery(query, sessionId || undefined);

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
    [sessionId]
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
