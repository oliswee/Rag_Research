"use client";

import { ResponseDisplay } from "./response-display";
import type { QueryResponse } from "@/types";
import { Loader2, MessageSquare, User } from "lucide-react";
import { useEffect, useRef } from "react";

export interface Message {
  type: "user" | "assistant";
  content: string;
  response?: QueryResponse;
  timestamp: Date;
}

export interface ConversationHistoryProps {
  messages: Message[];
  loading?: boolean;
  onFollowUp?: (suggestion: string) => void;
}

export function ConversationHistory({
  messages,
  loading,
  onFollowUp,
}: ConversationHistoryProps) {
  const endRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <MessageSquare className="mx-auto h-12 w-12 text-muted-foreground/50" />
          <p className="mt-4 text-muted-foreground">
            Start a conversation by asking a question about your research.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-6 overflow-y-auto">
      {messages.map((message, i) => (
        <div key={i} className="space-y-4">
          {message.type === "user" ? (
            <UserMessage content={message.content} />
          ) : (
            <AssistantMessage
              content={message.content}
              response={message.response}
              onFollowUp={onFollowUp}
            />
          )}
        </div>
      ))}

      {/* Loading state */}
      {loading && (
        <div className="flex items-start gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
            <MessageSquare className="h-4 w-4" />
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-muted p-4">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm text-muted-foreground">Thinking...</span>
          </div>
        </div>
      )}

      <div ref={endRef} />
    </div>
  );
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
        <User className="h-4 w-4" />
      </div>
      <div className="rounded-lg bg-muted p-4">
        <p className="text-sm">{content}</p>
      </div>
    </div>
  );
}

function AssistantMessage({
  content,
  response,
  onFollowUp,
}: {
  content: string;
  response?: QueryResponse;
  onFollowUp?: (suggestion: string) => void;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <MessageSquare className="h-4 w-4" />
      </div>
      <div className="flex-1 rounded-lg border bg-card p-4">
        {response ? (
          <ResponseDisplay response={response} onFollowUp={onFollowUp} />
        ) : (
          <p className="text-sm">{content}</p>
        )}
      </div>
    </div>
  );
}
