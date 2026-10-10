"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Search, Send } from "lucide-react";
import { useCallback, useState, type KeyboardEvent } from "react";

export interface SearchInputProps {
  onSubmit: (query: string) => void;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export function SearchInput({
  onSubmit,
  loading,
  disabled,
  placeholder = "Ask a question about your research...",
}: SearchInputProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = useCallback(() => {
    const trimmed = query.trim();
    if (!trimmed || loading || disabled) return;
    onSubmit(trimmed);
    setQuery("");
  }, [query, loading, disabled, onSubmit]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <div className="flex gap-2">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={loading || disabled}
          className="min-h-[48px] resize-none pl-9 pr-4"
          rows={1}
        />
      </div>
      <Button
        onClick={handleSubmit}
        disabled={!query.trim() || loading || disabled}
        className="h-12 w-12 shrink-0"
      >
        {loading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Send className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}
