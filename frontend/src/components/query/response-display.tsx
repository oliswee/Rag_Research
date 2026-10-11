"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { QueryResponse, SourceReference } from "@/types";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import { Fragment, useMemo, useState } from "react";
import { PatternDetailModal } from "./pattern-detail-modal";

export interface ResponseDisplayProps {
  response: QueryResponse;
  onFollowUp?: (suggestion: string) => void;
}

const CONFIDENCE_VARIANTS: Record<string, "default" | "success" | "warning" | "secondary"> = {
  high: "success",
  medium: "warning",
  low: "secondary",
};

export function ResponseDisplay({ response, onFollowUp }: ResponseDisplayProps) {
  const [showAllSources, setShowAllSources] = useState(false);
  const [selectedPatternId, setSelectedPatternId] = useState<string | null>(null);
  const [selectedRelevanceScore, setSelectedRelevanceScore] = useState<number | null>(null);

  const handlePatternClick = (patternId: string, relevanceScore?: number | null) => {
    setSelectedPatternId(patternId);
    setSelectedRelevanceScore(relevanceScore ?? null);
  };

  const handleCloseModal = () => {
    setSelectedPatternId(null);
    setSelectedRelevanceScore(null);
  };

  // Parse [P:id] citations in the response text
  const parsedResponse = useMemo(() => {
    const citationPattern = /\[P:([^\]]+)\]/g;
    const parts: Array<{ type: "text" | "citation"; value: string }> = [];
    let lastIndex = 0;
    let match;

    while ((match = citationPattern.exec(response.response)) !== null) {
      // Add text before citation
      if (match.index > lastIndex) {
        parts.push({
          type: "text",
          value: response.response.slice(lastIndex, match.index),
        });
      }
      // Add citation
      parts.push({ type: "citation", value: match[1] });
      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < response.response.length) {
      parts.push({
        type: "text",
        value: response.response.slice(lastIndex),
      });
    }

    return parts;
  }, [response.response]);

  const visibleSources = showAllSources
    ? response.sources
    : response.sources.slice(0, 3);

  if (response.retrieval_mode) {
    const evidence = response.evidence || [];
    const parts = response.response.split(/(\[C:[^\]]+\])/g);
    return <div className="space-y-5">
      <div className="whitespace-pre-wrap text-sm leading-7">{parts.map((part, index) => {
        const match = /^\[C:([^\]]+)\]$/.exec(part);
        if (!match) return <Fragment key={index}>{part}</Fragment>;
        const source = evidence.find((item) => item.chunk_id === match[1]);
        return source ? <a key={index} href={`#chunk-${source.chunk_id}`} className="mx-1 rounded bg-emerald-100 px-1.5 py-0.5 text-xs font-semibold text-emerald-900 hover:underline">{source.title.slice(0, 24)}{source.page_start ? ` · p. ${source.page_start}` : source.content_scope === "abstract" ? " · abstract" : ""}</a> : null;
      })}</div>
      {response.rewrite_status === "applied" && response.retrieval_query && (
        <p className="rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
          Searched as: {response.retrieval_query}
        </p>
      )}
      {(response.rewrite_status === "unavailable" || response.rewrite_status === "failed") && (
        <p className="text-xs text-muted-foreground">
          Query rewrite was unavailable; search used the conversation context.
        </p>
      )}
      {evidence.length > 0 && <section className="space-y-3"><h3 className="text-sm font-semibold">Source evidence · {response.retrieval_mode}</h3>{evidence.map((source) => <article id={`chunk-${source.chunk_id}`} key={source.chunk_id} className="rounded-xl border bg-muted/30 p-4"><div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-sm font-semibold">{source.title}</h4><span className="text-xs font-semibold uppercase tracking-wide text-emerald-700">{source.content_scope === "abstract" ? "Abstract" : source.page_start ? `Page ${source.page_start}` : "Source text"}</span></div><details className="mt-3 text-sm leading-6 text-muted-foreground"><summary className="cursor-pointer border-l-2 border-emerald-600 pl-3">{source.excerpt.slice(0, 260)}{source.excerpt.length > 260 ? "… Read full excerpt" : ""}</summary>{source.excerpt.length > 260 && <blockquote className="mt-3 border-l-2 border-emerald-600 pl-3">{source.excerpt}</blockquote>}</details>{source.source_url && <a href={`${source.source_url}${source.page_start ? `#page=${source.page_start}` : ""}`} target="_blank" rel="noopener noreferrer" className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-emerald-700 hover:underline">Open source <ExternalLink className="h-3 w-3" /></a>}</article>)}</section>}
    </div>;
  }

  return (
    <div className="space-y-4">
      {/* Response text with parsed citations */}
      <div className="prose prose-sm max-w-none dark:prose-invert">
        {parsedResponse.map((part, i) =>
          part.type === "citation" ? (
            <CitationLink
              key={i}
              patternId={part.value}
              sources={response.sources}
              onExpand={() => setShowAllSources(true)}
              onPatternClick={handlePatternClick}
            />
          ) : (
            <Fragment key={i}>{part.value}</Fragment>
          )
        )}
      </div>

      {/* Confidence badge */}
      {response.confidence && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Confidence:</span>
          <Badge variant={CONFIDENCE_VARIANTS[response.confidence] || "secondary"}>
            {response.confidence}
          </Badge>
        </div>
      )}

      {/* Sources */}
      {response.sources.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-muted-foreground">
            Sources ({response.sources.length})
          </h4>
          <div className="space-y-2">
            {visibleSources.map((source) => (
              <SourceCard
                key={source.pattern_id}
                source={source}
                onClick={() => handlePatternClick(source.pattern_id, source.relevance_score)}
              />
            ))}
          </div>
          {response.sources.length > 3 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAllSources(!showAllSources)}
              className="w-full"
            >
              {showAllSources ? (
                <>
                  <ChevronUp className="h-4 w-4" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="h-4 w-4" />
                  Show {response.sources.length - 3} more
                </>
              )}
            </Button>
          )}
        </div>
      )}

      {/* Follow-up suggestions */}
      {response.follow_up_suggestions && response.follow_up_suggestions.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-muted-foreground">
            Follow-up questions
          </h4>
          <div className="flex flex-wrap gap-2">
            {response.follow_up_suggestions.map((suggestion, i) => (
              <Button
                key={i}
                variant="outline"
                size="sm"
                onClick={() => onFollowUp?.(suggestion)}
                className="h-auto whitespace-normal text-left"
              >
                {suggestion}
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Pattern detail modal */}
      <PatternDetailModal
        patternId={selectedPatternId}
        relevanceScore={selectedRelevanceScore}
        onClose={handleCloseModal}
      />
    </div>
  );
}

function CitationLink({
  patternId,
  sources,
  onExpand,
  onPatternClick,
}: {
  patternId: string;
  sources: SourceReference[];
  onExpand?: () => void;
  onPatternClick?: (patternId: string, relevanceScore?: number | null) => void;
}) {
  const source = sources.find((s) => s.pattern_id === patternId);

  const handleClick = () => {
    // Open the pattern detail modal
    if (onPatternClick) {
      onPatternClick(patternId, source?.relevance_score);
      return;
    }

    // Fallback: Expand sources and scroll to source card
    onExpand?.();
    setTimeout(() => {
      const element = document.getElementById(`source-${patternId}`);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "center" });
        element.classList.add("ring-2", "ring-primary");
        setTimeout(() => {
          element.classList.remove("ring-2", "ring-primary");
        }, 2000);
      }
    }, 100);
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="inline-flex cursor-pointer items-center rounded bg-primary/10 px-1.5 py-0.5 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
      title={source?.summary || `Pattern ${patternId}`}
    >
      <ExternalLink className="mr-1 h-3 w-3" />
      {patternId.slice(0, 8)}
    </button>
  );
}

function SourceCard({
  source,
  onClick,
}: {
  source: SourceReference;
  onClick?: () => void;
}) {
  return (
    <Card
      id={`source-${source.pattern_id}`}
      className="bg-muted/50 transition-all cursor-pointer hover:bg-muted/70"
      onClick={onClick}
    >
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-foreground">
              {source.pattern_id.slice(0, 8)}...
            </p>
            {source.summary && (
              <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                {source.summary}
              </p>
            )}
          </div>
          {source.relevance_score != null && (
            <Badge variant="outline" className="shrink-0">
              {Math.round(source.relevance_score * 100)}%
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
