"use client";

import { uploadPaper } from "@/lib/api";
import type { PaperType, PaperUploadResponse } from "@/types";
import { AlertCircle, FileText, Loader2, Type } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  FileUpload,
  MetadataForm,
  TextInput,
  type MetadataFormData,
} from "@/components/upload";

type InputMode = "file" | "text";

const INITIAL_METADATA: MetadataFormData = {
  title: "",
  authors: "",
  source: "",
  publishedDate: "",
  type: "research_paper",
};

export default function UploadPage() {
  const router = useRouter();

  // Input state
  const [inputMode, setInputMode] = useState<InputMode>("file");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [textContent, setTextContent] = useState("");
  const [metadata, setMetadata] = useState<MetadataFormData>(INITIAL_METADATA);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PaperUploadResponse | null>(null);

  const hasContent =
    inputMode === "file" ? selectedFile !== null : textContent.trim().length > 0;

  const handleUpload = useCallback(async () => {
    if (!hasContent) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();

      // Add content
      if (inputMode === "file" && selectedFile) {
        formData.append("file", selectedFile);
      } else if (inputMode === "text") {
        formData.append("text", textContent);
      }

      // Add metadata (only non-empty values)
      if (metadata.title.trim()) {
        formData.append("title", metadata.title.trim());
      }
      if (metadata.authors.trim()) {
        formData.append("authors", metadata.authors.trim());
      }
      if (metadata.source.trim()) {
        formData.append("source", metadata.source.trim());
      }
      if (metadata.publishedDate) {
        formData.append("published_date", metadata.publishedDate);
      }
      if (metadata.type) {
        formData.append("type", metadata.type);
      }

      const response = await uploadPaper(formData);
      setResult(response);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Upload failed. Please try again."
      );
    } finally {
      setIsUploading(false);
    }
  }, [hasContent, inputMode, selectedFile, textContent, metadata]);

  const handleReviewPatterns = useCallback(() => {
    if (result) {
      router.push(`/review/${result.paper_id}`);
    }
  }, [result, router]);

  const handleUploadAnother = useCallback(() => {
    setSelectedFile(null);
    setTextContent("");
    setMetadata(INITIAL_METADATA);
    setResult(null);
    setError(null);
  }, []);

  // Show result after successful upload
  if (result) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-green-600 dark:text-green-400">
              <FileText className="h-5 w-5" />
              Upload Successful
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Extracted Metadata */}
            <div className="space-y-2">
              <h3 className="font-medium">Paper Details</h3>
              <dl className="space-y-1 text-sm">
                <div className="flex gap-2">
                  <dt className="text-muted-foreground">Title:</dt>
                  <dd className="font-medium">{result.metadata.title}</dd>
                </div>
                {result.metadata.authors.length > 0 && (
                  <div className="flex gap-2">
                    <dt className="text-muted-foreground">Authors:</dt>
                    <dd>{result.metadata.authors.join(", ")}</dd>
                  </div>
                )}
                <div className="flex gap-2">
                  <dt className="text-muted-foreground">Type:</dt>
                  <dd>{formatPaperType(result.metadata.type as PaperType)}</dd>
                </div>
              </dl>
            </div>

            {/* Pattern Count */}
            <div className="rounded-lg bg-primary/10 p-4">
              <p className="text-lg font-medium">
                {result.patterns.length} pattern
                {result.patterns.length !== 1 ? "s" : ""} extracted
              </p>
              <p className="text-sm text-muted-foreground">
                Review and approve patterns to add them to your knowledge base.
              </p>
            </div>

            {/* Warnings */}
            {result.extraction_warnings.length > 0 && (
              <div className="space-y-2 rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-900 dark:bg-yellow-950">
                <h4 className="flex items-center gap-2 font-medium text-yellow-800 dark:text-yellow-200">
                  <AlertCircle className="h-4 w-4" />
                  Warnings
                </h4>
                <ul className="space-y-1 text-sm text-yellow-700 dark:text-yellow-300">
                  {result.extraction_warnings.map((warning, i) => (
                    <li key={i}>• {warning}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3">
              <Button onClick={handleReviewPatterns} className="flex-1">
                Review Patterns
              </Button>
              <Button
                variant="outline"
                onClick={handleUploadAnother}
              >
                Upload Another
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-foreground">Upload Paper</h1>
        <p className="mt-1 text-muted-foreground">
          Upload a PDF or paste text to extract research patterns.
        </p>
      </div>

      <div className="space-y-6">
        {/* Input Mode Toggle */}
        <div className="flex gap-2 rounded-lg border border-input bg-card p-1">
          <button
            type="button"
            onClick={() => setInputMode("file")}
            className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              inputMode === "file"
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <FileText className="h-4 w-4" />
            PDF Upload
          </button>
          <button
            type="button"
            onClick={() => setInputMode("text")}
            className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              inputMode === "text"
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Type className="h-4 w-4" />
            Text Input
          </button>
        </div>

        {/* Content Input */}
        {inputMode === "file" ? (
          <FileUpload
            selectedFile={selectedFile}
            onFileSelect={setSelectedFile}
            onFileRemove={() => setSelectedFile(null)}
            disabled={isUploading}
          />
        ) : (
          <TextInput
            value={textContent}
            onChange={setTextContent}
            disabled={isUploading}
          />
        )}

        {/* Metadata Form */}
        <MetadataForm
          data={metadata}
          onChange={setMetadata}
          disabled={isUploading}
        />

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Upload Button */}
        <Button
          onClick={handleUpload}
          disabled={!hasContent || isUploading}
          className="w-full"
          size="lg"
        >
          {isUploading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            "Upload & Extract Patterns"
          )}
        </Button>

        <p className="text-center text-xs text-muted-foreground">
          The system will extract structured patterns (Claim/Evidence/Context)
          from your content using AI.
        </p>
      </div>
    </div>
  );
}

function formatPaperType(type: PaperType): string {
  return type
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
