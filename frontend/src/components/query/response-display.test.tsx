import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { ResponseDisplay } from "./response-display";

it("labels temporary Web evidence and bibliographic records without invented pages", () => {
  render(
    <ResponseDisplay
      response={{
        session_id: "s",
        response: "Unknown experiments.",
        sources: [],
        confidence: "low",
        follow_up_suggestions: [],
        retrieval_mode: "hybrid",
        routing: {
          intent: "latest",
          action: "search_web",
          sufficiency: "uncertain",
          reasons: ["freshness_required"],
          local_retry_count: 0,
          local_only: false,
          only_uploaded: false,
        },
        notices: ["Web evidence was not imported."],
        evidence: [
          {
            paper_id: "p",
            chunk_id: "web:1",
            title: "Abstract study",
            source_url: "https://example.org/a",
            content_scope: "abstract",
            excerpt: "Memory accuracy improves.",
            page_start: null,
            page_end: null,
            source_origin: "web",
            evidence_level: "abstract_only",
            license: "cc-by",
            retrieved_at: "2026-10-25T00:00:00Z",
          },
        ],
        web_records: [
          {
            paper_id: "m",
            title: "Metadata study",
            source_url: "https://example.org/m",
            evidence_level: "metadata_only",
            license: null,
            retrieved_at: "2026-10-25T00:00:00Z",
          },
        ],
      }}
    />
  );
  expect(screen.getByText("Web · Abstract only")).toBeInTheDocument();
  expect(screen.getByText("Metadata study")).toBeInTheDocument();
  expect(
    screen.getByText(/bibliographic information only/i)
  ).toBeInTheDocument();
  expect(
    screen.getByText("Web evidence was not imported.")
  ).toBeInTheDocument();
  expect(screen.queryByText(/Page \d/)).not.toBeInTheDocument();
});

it("does not render unsafe provider URLs as links", () => {
  render(
    <ResponseDisplay
      response={{
        session_id: "s",
        response: "Unknown.",
        sources: [],
        confidence: "low",
        follow_up_suggestions: [],
        retrieval_mode: "hybrid",
        evidence: [
          {
            paper_id: "p",
            chunk_id: "web:2",
            title: "Unsafe",
            source_url: "javascript:alert(1)",
            content_scope: "abstract",
            excerpt: "An abstract.",
            page_start: null,
            page_end: null,
          },
        ],
      }}
    />
  );
  expect(
    screen.queryByRole("link", { name: /open source/i })
  ).not.toBeInTheDocument();
});

it("shows the effective search query when rewriting was applied", () => {
  render(
    <ResponseDisplay
      response={{
        session_id: "session-1",
        response: "A cited answer.",
        sources: [],
        confidence: "medium",
        follow_up_suggestions: [],
        evidence: [],
        retrieval_mode: "bm25",
        retrieval_query: "active retrieval latency",
        rewrite_status: "applied",
      }}
    />
  );

  expect(screen.getByText(/searched as/i)).toHaveTextContent(
    "active retrieval latency"
  );
});
