"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { PaperResponse, PatternResponse, PatternUpdateRequest } from "@/types";
import { deletePattern, listPatterns, updatePattern } from "@/lib/api";
import {
  AlertCircle,
  Edit2,
  Loader2,
  Save,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

export interface PatternDetailProps {
  paper: PaperResponse;
  onClose: () => void;
  onPatternsChange: () => void;
}

export function PatternDetail({
  paper,
  onClose,
  onPatternsChange,
}: PatternDetailProps) {
  const [patterns, setPatterns] = useState<PatternResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Edit state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editData, setEditData] = useState<PatternUpdateRequest>({});
  const [saving, setSaving] = useState(false);

  // Delete state
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Load patterns
  useEffect(() => {
    async function loadPatterns() {
      try {
        setLoading(true);
        setError(null);
        const response = await listPatterns(paper.id);
        setPatterns(response.patterns);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load patterns");
      } finally {
        setLoading(false);
      }
    }

    loadPatterns();
  }, [paper.id]);

  // Start editing
  const handleEdit = useCallback((pattern: PatternResponse) => {
    setEditingId(pattern.id);
    setEditData({
      name: pattern.name,
      claim: pattern.claim || "",
      evidence: pattern.evidence || "",
      context: pattern.context || "",
      tags: pattern.tags,
    });
  }, []);

  // Cancel editing
  const handleCancelEdit = useCallback(() => {
    setEditingId(null);
    setEditData({});
  }, []);

  // Save edit
  const handleSave = useCallback(async () => {
    if (!editingId) return;

    setSaving(true);
    try {
      const updated = await updatePattern(editingId, editData);
      setPatterns((prev) =>
        prev.map((p) => (p.id === editingId ? updated : p))
      );
      setEditingId(null);
      setEditData({});
      onPatternsChange();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save pattern");
    } finally {
      setSaving(false);
    }
  }, [editingId, editData, onPatternsChange]);

  // Delete pattern
  const handleDelete = useCallback(
    async (patternId: string) => {
      setDeletingId(patternId);
      try {
        await deletePattern(patternId);
        setPatterns((prev) => prev.filter((p) => p.id !== patternId));
        onPatternsChange();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to delete pattern");
      } finally {
        setDeletingId(null);
      }
    },
    [onPatternsChange]
  );

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-lg flex-col border-l border-input bg-background shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-input p-4">
        <div className="min-w-0 flex-1">
          <h2 className="truncate font-semibold text-foreground">
            {paper.title}
          </h2>
          <p className="text-sm text-muted-foreground">
            {patterns.length} pattern{patterns.length !== 1 ? "s" : ""}
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Error */}
        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
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

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        )}

        {/* Empty state */}
        {!loading && patterns.length === 0 && (
          <p className="py-8 text-center text-muted-foreground">
            No patterns found for this paper.
          </p>
        )}

        {/* Pattern list */}
        {!loading && patterns.length > 0 && (
          <div className="space-y-4">
            {patterns.map((pattern) => (
              <div
                key={pattern.id}
                className="rounded-lg border border-input bg-card p-4"
              >
                {editingId === pattern.id ? (
                  // Edit mode
                  <div className="space-y-3">
                    <Input
                      label="Name"
                      value={editData.name || ""}
                      onChange={(e) =>
                        setEditData({ ...editData, name: e.target.value })
                      }
                      disabled={saving}
                    />
                    <Textarea
                      label="Claim"
                      value={editData.claim || ""}
                      onChange={(e) =>
                        setEditData({ ...editData, claim: e.target.value })
                      }
                      disabled={saving}
                      className="min-h-[80px]"
                    />
                    <Textarea
                      label="Evidence"
                      value={editData.evidence || ""}
                      onChange={(e) =>
                        setEditData({ ...editData, evidence: e.target.value })
                      }
                      disabled={saving}
                      className="min-h-[80px]"
                    />
                    <Textarea
                      label="Context"
                      value={editData.context || ""}
                      onChange={(e) =>
                        setEditData({ ...editData, context: e.target.value })
                      }
                      disabled={saving}
                      className="min-h-[80px]"
                    />
                    <Input
                      label="Tags (comma-separated)"
                      value={editData.tags?.join(", ") || ""}
                      onChange={(e) =>
                        setEditData({
                          ...editData,
                          tags: e.target.value
                            .split(",")
                            .map((t) => t.trim())
                            .filter(Boolean),
                        })
                      }
                      disabled={saving}
                    />
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        onClick={handleSave}
                        disabled={saving}
                      >
                        {saving ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Save className="h-4 w-4" />
                        )}
                        Save
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleCancelEdit}
                        disabled={saving}
                      >
                        <X className="h-4 w-4" />
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  // View mode
                  <>
                    <div className="flex items-start justify-between gap-2">
                      <h4 className="font-medium text-foreground">
                        {pattern.name}
                      </h4>
                      <div className="flex gap-1">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleEdit(pattern)}
                          disabled={deletingId === pattern.id}
                        >
                          <Edit2 className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleDelete(pattern.id)}
                          disabled={deletingId === pattern.id}
                          className="text-destructive hover:bg-destructive/10"
                        >
                          {deletingId === pattern.id ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <Trash2 className="h-3.5 w-3.5" />
                          )}
                        </Button>
                      </div>
                    </div>

                    {/* Tags and status */}
                    <div className="mt-2 flex flex-wrap gap-1">
                      <Badge
                        variant={
                          pattern.status === "approved" ? "success" : "secondary"
                        }
                      >
                        {pattern.status}
                      </Badge>
                      {pattern.tags.map((tag) => (
                        <Badge key={tag} variant="outline">
                          {tag}
                        </Badge>
                      ))}
                    </div>

                    {/* Content */}
                    <div className="mt-3 space-y-2 text-sm">
                      {pattern.claim && (
                        <div>
                          <span className="font-medium text-muted-foreground">
                            Claim:{" "}
                          </span>
                          <span className="text-foreground">{pattern.claim}</span>
                        </div>
                      )}
                      {pattern.evidence && (
                        <div>
                          <span className="font-medium text-muted-foreground">
                            Evidence:{" "}
                          </span>
                          <span className="text-foreground">
                            {pattern.evidence}
                          </span>
                        </div>
                      )}
                      {pattern.context && (
                        <div>
                          <span className="font-medium text-muted-foreground">
                            Context:{" "}
                          </span>
                          <span className="text-foreground">
                            {pattern.context}
                          </span>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
