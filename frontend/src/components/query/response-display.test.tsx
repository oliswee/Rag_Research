import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { ResponseDisplay } from "./response-display";

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
    />,
  );

  expect(screen.getByText(/searched as/i)).toHaveTextContent(
    "active retrieval latency",
  );
});
