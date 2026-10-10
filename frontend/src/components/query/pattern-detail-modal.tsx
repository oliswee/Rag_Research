"use client";

import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { LoadingSpinner } from "@/components/ui/loading";
import { getPattern } from "@/lib/api";
import type { PatternDetailResponse } from "@/types";
import { BookOpen, FileText, Lightbulb, MessageSquareQuote } from "lucide-react";
import { useEffect, useState } from "react";

export interface PatternDetailModalProps {
  patternId: string | null;
  relevanceScore?: number | null;
  onClose: () => void;
}

export function PatternDetailModal({
  patternId,
  relevanceScore,
  onClose,
}: PatternDetailModalProps) {
  const [pattern, setPattern] = useState<PatternDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!patternId) {
      setPattern(null);
      setError(null);
      return;
    }

    async function fetchPattern() {
      setLoading(true);
      setError(null);
      try {
        const data = await getPattern(patternId!);
        setPattern(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load pattern");
      } finally {
        setLoading(false);
      }
    }

    fetchPattern();
  }, [patternId]);

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      onClose();
    }
  };

  return (
    <Dialog open={!!patternId} onOpenChange={handleOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-2xl overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" />
          </div>
        ) : error ? (
          <div className="py-8 text-center">
            <p className="text-destructive">{error}</p>
          </div>
        ) : pattern ? (
          <>
            <DialogHeader>
              <div className="flex items-start justify-between gap-4 pr-8">
                <div className="space-y-1">
                  <DialogTitle className="text-xl">{pattern.name}</DialogTitle>
                  <DialogDescription className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    {pattern.paper_title}
                  </DialogDescription>
                </div>
                {relevanceScore != null && (
                  <Badge variant="outline" className="shrink-0">
                    {Math.round(relevanceScore * 100)}% match
                  </Badge>
                )}
              </div>
            </DialogHeader>

            <div className="mt-4 space-y-6">
              {/* Claim */}
              {pattern.claim && (
                <section className="space-y-2">
                  <h3 className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <Lightbulb className="h-4 w-4" />
                    Claim
                  </h3>
                  <p className="text-sm leading-relaxed">{pattern.claim}</p>
                </section>
              )}

              {/* Evidence */}
              {pattern.evidence && (
                <section className="space-y-2">
                  <h3 className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <MessageSquareQuote className="h-4 w-4" />
                    Evidence
                  </h3>
                  <blockquote className="border-l-2 border-muted-foreground/20 pl-4 text-sm italic leading-relaxed text-muted-foreground">
                    {pattern.evidence}
                  </blockquote>
                </section>
              )}

              {/* Context */}
              {pattern.context && (
                <section className="space-y-2">
                  <h3 className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                    <BookOpen className="h-4 w-4" />
                    Context
                  </h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {pattern.context}
                  </p>
                </section>
              )}

              {/* Tags */}
              {pattern.tags && pattern.tags.length > 0 && (
                <section className="space-y-2">
                  <h3 className="text-sm font-medium text-muted-foreground">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {pattern.tags.map((tag) => (
                      <Badge key={tag} variant="secondary">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </section>
              )}
            </div>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
